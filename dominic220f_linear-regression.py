#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue Sep  5 15:25:08 2017



@author: dom

"""

# importing the libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

from sklearn.preprocessing import Imputer

from sklearn.linear_model import LinearRegression

import math

# importing the dataset

train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

train = train.dropna()

test = test.dropna()



X_train = train.iloc[:,0:1].values

y_train = train.iloc[:,1:].values

X_test = test.iloc[:,0:1].values

y_test = test.iloc[:,1:].values



# Fitting the model to the training dataset

regressor = LinearRegression()

regressor.fit(X_train,y_train)



# Printing R sq and Correlation

print('R square = ',regressor.score(X_train,y_train))

print('Correlation = ',math.sqrt(regressor.score(X_train,y_train)))



# Predicting the Y values

y_pred = regressor.predict(X_train)



# Visualising the training set

plt.scatter(X_train,y_train, color = 'red')

plt.plot(X_train,regressor.predict(X_train), color ='blue')

plt.title('Training Set')

plt.show()



# Visualising the test set

plt.scatter(X_test,y_test, color = 'red')

plt.plot(X_train,regressor.predict(X_train), color ='blue')

plt.title('Test Set')

plt.show()








