# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 13:56:28 2020

@author: 91842
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Training data

df = pd.read_csv("/kaggle/input/titanic/train.csv")
df_test = pd.read_csv("/kaggle/input/titanic/test.csv")
df_output = pd.read_csv("/kaggle/input/titanic/gender_submission.csv")

cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Sex']
df = df[cols]
cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Sex']
df_test = df_test[cols]
df_test = df_test.reset_index()

X = df.iloc[:, 1:].values
y = df.iloc[:, 0].values

X_test = df_test.iloc[:,1:].values
y_test = df_output.iloc[:,1:].values
# Replace missing points
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:2])
X[:, 1:2] = imputer.transform(X[:,1:2]) 
X_test[:, [1,4]] = imputer.fit_transform(X_test[:,[1,4]])

imp = SimpleImputer(missing_values = np.nan, strategy = 'most_frequent')
X[:,[5,6]] = imp.fit_transform(X[:,[5,6]])
X_test[:,[5,6]] = imp.fit_transform(X_test[:, [5,6]])







# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [5,6])], remainder = 'passthrough')
X = np.array(ct.fit_transform(X))
X_test = np.array(ct.transform(X_test))



# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [6,9]] = sc.fit_transform(X[:, [6,9]])
X_test[:, [6,9]] = sc.transform(X_test[:, [6,9]])

 
# Learning the model Linear Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X, y) 

 
# # Learning the model K Nearest Neighbors
# from sklearn.neighbors import KNeighborsClassifier
# kNN = KNeighborsClassifier(n_neighbors = 10, algorithm = 'kd_tree', p =1)
# kNN.fit(X, y)  

# #Learning the model Support Vector Machine 
# from sklearn.svm import SVC
# svc = SVC(C = 0.01, kernel = 'linear' )
# svc.fit(X,y) 


y_predict = classifier.predict(X_test)
y_predict = y_predict.reshape(len(y_predict),1)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_predict)
