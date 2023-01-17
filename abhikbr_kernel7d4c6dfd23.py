# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression

from sklearn import preprocessing





# plotly library

import plotly.graph_objs as go

from plotly import tools

from plotly.offline import init_notebook_mode, iplot
data=pd.read_csv('../input/indian-liver-patient-records/indian_liver_patient.csv')

data.head()
data.info()
male=pd.get_dummies(data['Gender'],drop_first=True)

male.head()
data=pd.concat([data,male],axis=1)

data.drop('Gender',axis=1,inplace=True)

data.head()
data.isnull().sum()
data['Albumin_and_Globulin_Ratio']=data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean())

data.isnull().sum()
data.plot(kind='box',figsize=(25,10),subplots=True,layout=(3,4))
data.columns
sns.countplot(data['Male'],hue=data['Dataset'])
sns.barplot(data=data, x='Male', y='Total_Bilirubin', hue='Dataset')
sns.barplot(data=data, x='Male', y='Direct_Bilirubin', hue='Dataset')
plt.figure(figsize=(11,6))

sns.heatmap(data.corr(),annot=True,linewidths=1.5)
plt.figure(figsize=(11,6))

sns.scatterplot(data=data,x='Albumin',y='Total_Protiens',hue='Dataset')
plt.figure(figsize=(11,6))

sns.scatterplot(data=data,x='Albumin',y='Albumin_and_Globulin_Ratio',hue='Dataset')
data['all_albumin']=data['Albumin']+data['Albumin_and_Globulin_Ratio']

data['all_albumin'].corr(data['Dataset'])
data.columns
plt.figure(figsize=(11,6))

sns.scatterplot(data=data,x='Direct_Bilirubin',y='Total_Bilirubin',hue='Dataset')
data['total_bilrubin']=data['Direct_Bilirubin']-data['Total_Bilirubin']

data['total_bilrubin'].corr(data['Dataset'])
data.columns
drop_all=['Albumin','Albumin_and_Globulin_Ratio','Direct_Bilirubin','Total_Bilirubin']

data.drop(drop_all,axis=1,inplace=True)

data.columns
plt.figure(figsize=(11,6))

sns.lineplot(data=data,x='Alamine_Aminotransferase',y= 'Aspartate_Aminotransferase',hue='Dataset',palette="Paired")
data=data.drop(data[(data['Alamine_Aminotransferase']>1250) & (data['Aspartate_Aminotransferase']<3500)].index)
data.isnull().sum()
plt.figure(figsize=(11,6))

sns.heatmap(data.corr(),annot=True)
X=data.drop('Dataset',axis=1)

y=data['Dataset']
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)
from sklearn.neighbors import KNeighborsClassifier

from sklearn.ensemble import RandomForestClassifier

from sklearn.svm import SVC

from sklearn import tree





from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
svc=SVC(gamma=1,C=50)

scores_svc=cross_val_score(svc,X_train,y_train,cv=10,scoring='accuracy')

print(scores_svc)

print(scores_svc.mean())
score_svc1=svc.fit(X_train,y_train)
rfc=RandomForestClassifier(max_depth=1,max_features=6)

scores_rfc=cross_val_score(rfc,X_train,y_train,cv=10,scoring='accuracy')

print(scores_rfc)

print(scores_rfc.mean())
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import GridSearchCV

from scipy.stats import uniform
def get_best_score(model):

    

    print(model.best_score_)    

    print(model.best_params_)

    print(model.best_estimator_)

    

    return model.best_score_
model = SVC()

param_grid = {'C':uniform(0.01, 5000), 'gamma':uniform(0.0001, 1) }

rand_SVC = RandomizedSearchCV(model, param_distributions=param_grid, n_iter=100)

rand_SVC.fit(X_train,y_train)

score_rand_SVC = get_best_score(rand_SVC)
param_grid = {'C': [0.1,10, 100, 1000,5000], 'gamma': [1,0.1,0.01,0.001,0.0001], 'kernel': ['rbf']}

svc_grid = GridSearchCV(SVC(), param_grid, cv=10, refit=True, verbose=1)

svc_grid.fit(X_train,y_train)

sc_svc = get_best_score(svc_grid)
knn = KNeighborsClassifier()

leaf_range = list(range(3, 15, 1))

k_range = list(range(1, 15, 1))

weight_options = ['uniform', 'distance']

param_grid = dict(leaf_size=leaf_range, n_neighbors=k_range, weights=weight_options)

print(param_grid)



knn_grid = GridSearchCV(knn, param_grid, cv=10, verbose=1, scoring='accuracy')

knn_grid.fit(X_train, y_train)



sc_knn = get_best_score(knn_grid)
pred=score_svc1.predict(X_test)
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

print(confusion_matrix(pred,y_test))

print(classification_report(pred,y_test))