import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split



from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

from sklearn.neural_network import MLPClassifier
data = pd.read_csv("../input/mushroom-classification/mushrooms.csv")
data
mappings = list()



encoder = LabelEncoder()



for column in range(len(data.columns)):

    data[data.columns[column]] = encoder.fit_transform(data[data.columns[column]])

    mappings_dict = {index: label for index, label in enumerate(encoder.classes_)}

    mappings.append(mappings_dict)
mappings
y = data['class']

X = data.drop('class', axis=1)
scaler = StandardScaler()



X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
log_model = LogisticRegression()

svm_model = SVC(C=1.0, kernel='rbf')

nn_model = MLPClassifier(hidden_layer_sizes=(128, 128))
np.sum(y) / len(y)
log_model.fit(X_train, y_train)

svm_model.fit(X_train, y_train)

nn_model.fit(X_train, y_train)
print(f"---Logistic Regression: {log_model.score(X_test, y_test)}")

print(f"Support Vector Machine: {svm_model.score(X_test, y_test)}")

print(f"--------Neural Network: {nn_model.score(X_test, y_test)}")
X_test.shape
corr = data.corr()



sns.heatmap(corr)