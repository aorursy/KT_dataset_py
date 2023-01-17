import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from sklearn.datasets import load_breast_cancer

import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

from sklearn.model_selection import GridSearchCV
cancer = load_breast_cancer()
df_cancer = pd.DataFrame(np.c_[cancer['data'],cancer['target']], columns = np.append(cancer['feature_names'], ['target']))
sns.pairplot(df_cancer, vars = ['mean radius','mean texture','mean area','mean perimeter','mean smoothness'], hue = 'target')
sns.countplot(df_cancer['target'])
sns.scatterplot(x=df_cancer['mean radius'], y = df_cancer['mean area'], hue = 'target', data = df_cancer)
df_cancer
plt.figure(figsize= (20,20))

sns.heatmap(df_cancer.corr(), annot  = True)
x = df_cancer.drop('target', axis = 1)

y = df_cancer['target']



x_train,x_test, y_train,y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)
svm_model = SVC()

svm_model.fit(x_train, y_train)
y_predict = svm_model.predict(x_test)
y_predict
sns.scatterplot(x = x_train['mean area'], y = x_train['mean smoothness'], hue = y_train)
cm = confusion_matrix(y_test, y_predict)

sns.heatmap(cm , annot = True)

print(classification_report(y_test, y_predict))
param_grid = {'C': [0.1, 1,10,100], 'gamma':[1,0.1,0.01,0.001], 'kernel' : ['rbf']} 
grid = GridSearchCV(SVC(), param_grid, verbose=4, refit = True)

grid.fit(x_train, y_train)

grid_pred = grid.predict(x_test)
cm = confusion_matrix(y_test, grid_pred)

sns.heatmap(cm, annot = True)



print(classification_report(y_test, grid_pred))