import numpy as np

import pandas as pd 
import matplotlib.pyplot as plt

import seaborn as sns 
%matplotlib inline
data = pd.read_csv("../input/data.csv")
data.head()
data.info()
data.shape
data.describe()
data.isnull().sum(axis=0)
data=data.drop("Unnamed: 32",axis=1)
X = data.drop(["id","diagnosis"],axis=1)

y = data["diagnosis"]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.svm import SVC
# Instantiating SVM model (basically creating a svm object)
model = SVC()
# Training or fitting the model on training data
model.fit(X_train, y_train)
predictions = model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test,predictions))
print(confusion_matrix(y_test,predictions))
from sklearn.model_selection import GridSearchCV
param_grid = {'C':[0.1,1,10,100,1000],'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=3)
grid.fit(X_train, y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))