#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""

Created on Tue Sep  5 17:22:53 2017



@author: dom

"""



# importing the libraries

import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import math

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



# importing data set

dataset = pd.read_csv('../input/Height_Weight_single_variable_data_101_series_1.0.csv')

X = dataset.iloc[:,1:]

y = dataset.iloc[:,0:1]



# checking for null values if any

dataset.isnull().any()



# Splitting the dataset into test set and training set

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1/3,random_state = 0)



# Fitting the Model 

regressor = LinearRegression()

regressor.fit(X_train,y_train)



# Calculation R square and Correlation

print('R square = ',regressor.score(X_train,y_train))

print('Correlation = ',math.sqrt(regressor.score(X_train,y_train)))



# Prediciting Height

y_pred = regressor.predict(X_train)



# Visualising the Training set

plt.scatter(X_train,y_train,color='red')

plt.plot(X_train, regressor.predict(X_train),color='blue')

plt.title('Training Set')

plt.xlabel('Weight')

plt.ylabel('Height')

plt.show()



# Visualising the trained model and testing it on our test set

plt.scatter(X_test,y_test,color='red')

plt.plot(X_train, regressor.predict(X_train),color='blue')

plt.title('Test Set')

plt.xlabel('Weight')

plt.ylabel('Height')

plt.show()



# Predicting the weight for any given height

print('For a person having height as 163 his predicted weight will be:',regressor.predict(166))




