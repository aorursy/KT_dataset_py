import numpy as np 

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

cancer = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
cancer.keys()
print(cancer['diagnosis'])
cancer['diagnosis'] = cancer['diagnosis'].map({'M':0, 'B':1})
del cancer['Unnamed: 32']
print(cancer['diagnosis'])
del cancer['id']
cancer.head()
sns.pairplot(cancer,hue='diagnosis',vars=['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean'])
sns.countplot(cancer['diagnosis'])
sns.scatterplot(x='area_mean',y='smoothness_mean',hue='diagnosis',data=cancer)
plt.figure(figsize=(20,10))

sns.heatmap(cancer.corr(),annot = True)
X = cancer.drop(['diagnosis'],axis=1)
y= cancer['diagnosis']
from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)
svc_model = SVC()

svc_model.fit(X_train,y_train)

y_pred=svc_model.predict(X_test)
y_pred
cm= confusion_matrix(y_test,y_pred)

sns.heatmap(cm,annot= True)
min_train = X_train.min()

range_train =(X_train-min_train).max()

X_train_scaled=(X_train -min_train)/range_train
sns.scatterplot(x=X_train['area_mean'],y=X_train['smoothness_mean'],hue=y_train)
sns.scatterplot(x=X_train_scaled['area_mean'],y=X_train_scaled['smoothness_mean'],hue=y_train)
min_test = X_test.min()

range_test =(X_test-min_test).max()

X_test_scaled =(X_test -min_test)/range_test
svc_model.fit(X_train_scaled,y_train)

y_predict=svc_model.predict(X_test_scaled)
cm = confusion_matrix(y_test,y_predict)

sns.heatmap(cm,annot=True)
print(classification_report(y_test,y_predict))
from sklearn.model_selection import GridSearchCV

param_grid={'C':[0.1,1,10,100], 'gamma': [1,1, 0.1,.001],'kernel':['rbf']}

grid=GridSearchCV(SVC(),param_grid, refit=True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_params_
grid_pred= grid.predict(X_test_scaled)

cm=confusion_matrix(y_test,grid_pred)
sns.heatmap(cm,annot=True)
print(classification_report(y_test,grid_pred))