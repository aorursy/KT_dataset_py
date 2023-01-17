# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/heart.csv')
data.info()
data.head()
plt.figure(figsize=(14,6))

sns.set_style('whitegrid')

sns.countplot(x='target',data=data)
plt.figure(figsize=(14,6))

sns.set_style('dark')

sns.countplot(x='target',hue='sex',data=data,palette='RdBu_r')
plt.figure(figsize=(14,6))

sns.set_style('dark')

sns.countplot(x='target',hue='thal',data=data)
data['age'].plot(kind='hist',bins=30,color='red',figsize= (16,7))
X=data.drop(columns=['target'],axis=1)

y=data['target']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=101)
from sklearn.linear_model import LogisticRegression
lr_model= LogisticRegression()

lr_model.fit(X_train,y_train)
lr_pred=lr_model.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_test,lr_pred))
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train,y_train)
svc_pred = svc_model.predict(X_test)
print(classification_report(y_test,svc_pred))
param_grid = {'C': [0.1,1, 10, 100, 1000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']} 
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=3)
# May take awhile!

grid.fit(X_train,y_train)
grid.best_params_
grid.best_estimator_
grid_predictions = grid.predict(X_test)
print(classification_report(y_test,grid_predictions))