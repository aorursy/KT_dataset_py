# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_diabetes

import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn import neighbors
#Load data into dataframe

diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data)
#Check null values in columns

df.isnull().any()



#Fill in NaN values

df.fillna(value=0,axis=1,inplace=True)
#Create correlation heatmap

sns.heatmap(df.corr(),vmax=1,square=True)
#Define Features and Target

features = [2,3]

target = [4]
#Split data into Training and Test sets

train, test = train_test_split(df,test_size=0.3)

train.head()
#Fill training and test data with necessary data

X_train = train[features].dropna()

y_train = train[target].dropna()

X_test = test[features].dropna()

y_test = test[target].dropna()
#Train model on traning set and run scores

linreg = LinearRegression()

linreg.fit(X_train,y_train)



lin_score_train = linreg.score(X_test, y_test)

lin_score_test = linreg.score(X_train, y_train)



#Print Rsquared values for training and test data

print("Training score: ", lin_score_train)

print("Testing score: ", lin_score_test)
#KNR Analysis

n_neighbors = 5

knn = neighbors.KNeighborsRegressor(n_neighbors, weights='uniform')

knn.fit(X_train,y_train)



#KNR scores

knn_score_train = knn.score(X_test, y_test)

knn_score_test = knn.score(X_train, y_train)



print("Training score: ", knn_score_train)

print("Testing score: ", knn_score_test)