import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()

%matplotlib inline
iris = pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
iris.head()
sns.pairplot(iris, hue='species')
iris_setosa = iris[iris['species'] == 'Iris-setosa']

iris_setosa.head()
sns.jointplot(x='sepal_width', y='sepal_length', data=iris_setosa, kind='kde')

plt.show()
# Splitting data into training and testing datasets

from sklearn.model_selection import train_test_split
X = iris.drop('species', axis=1)

y = iris['species']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=5)
# Training the model

from sklearn.svm import SVC
svm_classifier = SVC(gamma='auto')

svm_classifier.fit(X_train, y_train)
# Model Evaluation

from sklearn.metrics import confusion_matrix, classification_report
y_pred = svm_classifier.predict(X_test)
print(confusion_matrix(y_test, y_pred))

print(classification_report(y_test, y_pred))
svm_classifier.score(X_train, y_train)
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid_pred = grid.predict(X_test)
print(confusion_matrix(y_test, grid_pred))

print(classification_report(y_test, grid_pred))