

# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split







import os

data = pd.read_csv("../input/heart-disease-uci/heart.csv")



data.info()

data.head()
data.describe()

data.drop_duplicates(inplace=True)

print(data['age'].describe()) # No noise , all Value in Range .

print(data.corr())

print(data.corr()['target'].sort_values())
data = data.drop('oldpeak', axis = 1) 

print(data.isnull().sum()) #Checking for null values

from sklearn.model_selection import train_test_split

X = data[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','slope','ca','thal']]

y = data['target']
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

X_train

X_test

y_train

y_test 
print (X_train)

print (X_test)

print (y_train)

print (y_test)

print('X_train is ' , X_train.shape)

print('X_test is ' , X_test.shape)

print('y_train  is ' , y_train.shape)

print('y_test is ' , y_test.shape)
cont_data=data.copy()

from scipy.stats import pearsonr

from scipy import stats
stats.pearsonr(X['age'],y)

stats.pearsonr(X['sex'],y)

stats.pearsonr(X['cp'],y)

stats.pearsonr(X['trestbps'],y)
