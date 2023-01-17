# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

#from sklearn.datasets import  load_breast_cancer

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC

from sklearn.preprocessing import MinMaxScaler



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
cancer=pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')

cancer.head()
print(cancer.shape)
#Mapping categorical data to  numerical data

cancer['diagnosis']=cancer['diagnosis'].map({'M':1,'B':0})



y = cancer.diagnosis 



l= ['Unnamed: 32','id','diagnosis']

X = cancer.drop(l,axis=1)#remove the unwanted column and target column

X.head()

print(y)
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0)
print(X_train.shape)

print(X_test.shape)
clf = SVC(C=10).fit(X_train, y_train)

print(clf.score(X_train, y_train))

print(clf.score(X_test, y_test))
scaler=MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

clf = SVC(C=10).fit(X_train, y_train)

print(clf.score(X_train_scaled, y_train))

print(clf.score(X_test_scaled, y_test))
from sklearn.model_selection import GridSearchCV

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.001,0.0001],

                     'C': [1, 10, 100, 1000]},

                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

grid=GridSearchCV(SVC(),tuned_parameters,cv=5)

grid.fit(X_train_scaled,y_train)

grid.best_params_
grid.best_score_