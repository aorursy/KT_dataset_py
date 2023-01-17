import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('../input/data.csv')
data.head()
sns.pairplot(data, hue = 'diagnosis', vars = ['radius_mean', 'texture_mean', 'area_mean', 'perimeter_mean', 'smoothness_mean'])
sns.countplot(data.diagnosis)
sns.scatterplot(data = data, x = 'area_mean', y = 'smoothness_mean', hue = 'diagnosis')
plt.figure(figsize = (20 ,10))
sns.heatmap(data.corr(), annot = True)
X = data.drop(['id', 'diagnosis', 'Unnamed: 32'], axis = 1)
y = data['diagnosis']
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)
X_train
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
svc_model = SVC()
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm
(cm[0][0] + cm[1][1]) / 114
sns.heatmap(cm, annot = True)
X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min())
X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min())
sns.scatterplot(data = X_train_scaled,  x = 'area_mean', y = 'smoothness_mean', hue = y_train)
svc_model.fit(X_train_scaled, y_train)
y_pred = svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)
cm
sns.heatmap(cm, annot = True)
print(classification_report(y_test, y_pred))
param_grid = { 'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 4)
grid.fit(X_train_scaled, y_train)
grid.best_params_
grid_pred = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test, grid_pred)
sns.heatmap(cm, annot = True)
print(classification_report(y_test, grid_pred))
