import numpy as np
import matplotlib.pyplot as plt
from scipy.io import arff
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder


def load_arff_data(file_path):
    """Carrega e processa os dados do ARFF, convertendo valores categóricos em numéricos."""
    data, meta = arff.loadarff(file_path)
    
    features = []
    encoders = {} 

    for attr in meta.names()[:-1]: 
        col = data[attr]

        
        if meta[attr][0] == 'nominal':
            col = np.array([x.decode('utf-8').strip() for x in col])
            le = LabelEncoder()
            col = le.fit_transform(col)
            encoders[attr] = le 

        features.append(col)

    features = np.column_stack(features)

   
    target = np.array([x.decode('utf-8').strip() for x in data[meta.names()[-1]]])
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(target)

    return features, target, meta.names()[:-1], target_encoder, encoders


def train_decision_tree(features, target):
   
    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
    model.fit(features, target)
    return model


def plot_decision_tree(model, feature_names, class_names):
    
    plt.figure(figsize=(12, 7))
    plot_tree(model, feature_names=feature_names, class_names=class_names, filled=True, rounded=True)
    plt.show()


def plot_confusion_matrix(model, features, target, class_names):
    
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics.ConfusionMatrixDisplay.from_estimator(
        model, features, target, display_labels=class_names, values_format='d', ax=ax
    )
    plt.show()


if __name__ == "__main__":
    file_path = "./bank.arff"

    
    features, target, feature_names, target_encoder, encoders = load_arff_data(file_path)

   
    model = train_decision_tree(features, target)

    
    plot_decision_tree(model, feature_names, target_encoder.classes_)

    
    plot_confusion_matrix(model, features, target, target_encoder.classes_)
