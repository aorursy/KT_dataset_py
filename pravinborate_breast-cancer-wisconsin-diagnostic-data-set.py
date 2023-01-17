# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
data.head()
data.info()
data.isnull().sum()
data.drop('Unnamed: 32',inplace = True,axis = 1)
data.head()
data.shape
data.info()
sns.pairplot(data,hue = 'diagnosis',vars = ['radius_mean','texture_mean','perimeter_mean','area_mean','smoothness_mean'])
sns.countplot(data['diagnosis'])
sns.scatterplot(x='area_mean',y='smoothness_mean',hue = 'diagnosis',data=data)
data_dict = {

    'M':0,

    'B':1

}



data['diagnosis'] = data['diagnosis'].replace(data_dict)
data.drop('id',inplace = True,axis = 1)
plt.figure(figsize=(25,20))

sns.heatmap(data.corr(),annot=True)
X = data.drop(['diagnosis'],axis = 1)
y = data['diagnosis']
X.head()
y.value_counts()
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 42)
print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
svc_model = SVC()
svc_model.fit(X_train,y_train)
y_pred = svc_model.predict(X_test)
y_pred
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot = True)
min_train = X_train.min()

print(min_train)
range_train = (X_train-min_train).max()

print(range_train)
X_train_scaled = (X_train - min_train)/range_train
sns.scatterplot(x=X_train['area_mean'],y=X_train['smoothness_mean'],hue = y_train)
sns.scatterplot(x = X_train_scaled['area_mean'],y=X_train_scaled['smoothness_mean'],hue = y_train)
min_test = X_test.min()

range_test = (X_test - min_test).max()

X_test_scaled = (X_test - min_test)/range_test
svc_model.fit(X_train_scaled,y_train)
y_pred = svc_model.predict(X_test_scaled)
print(y_pred)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot = True)
print(classification_report(y_test,y_pred))
param_grid = {'C':[0.1,1,10,100],'gamma':[1,0.1,0.01,0.001],'kernel':['rbf']}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(SVC(),param_grid,refit = True,verbose=4)
grid.fit(X_train_scaled,y_train)
grid.best_estimator_
grid_prediction = grid.predict(X_test_scaled)
cm = confusion_matrix(y_test,grid_prediction)
sns.heatmap(cm,annot = True)
print(classification_report(y_test,grid_prediction))