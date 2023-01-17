#House Price Prediction using Linear Regression

#Libraries Import

import numpy as np # linear algebra 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

from sklearn import svm

from sklearn import linear_model

from sklearn import metrics

from sklearn.preprocessing import Imputer

import matplotlib.pyplot as plt

import seaborn as sns





print(os.listdir("../input"))



dataframe=pd.read_csv('../input/kc_house_data.csv')



dataframe=dataframe.drop(['id', 'date'], axis=1)



#dataframe.isnull().values.any()



#dataframe['waterfront'].unique()

dataframe.head(5)
dataframe.dropna(axis='rows')
x=dataframe[['bedrooms','bathrooms','sqft_living','sqft_lot','floors','waterfront','view','condition','grade','sqft_above','sqft_basement','yr_built','yr_renovated','zipcode','lat','long','sqft_living15','sqft_lot15']]

y=dataframe['price']

print(x.head())

print(y.head())

#train and test dataset creation

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)



#Creating a linear regression model

regression = linear_model.LinearRegression()

regression.fit(X_train,y_train)

predicted_Values = regression.predict(X_test)



#Checking accuracy of matrix

mean_squared_error = metrics.mean_squared_error(y_test, predicted_Values)

print('Mean Squared Error (MSE) ', round(np.sqrt(mean_squared_error), 2)) #239902,219742 (predicted value - actual value of y)/(total number of values predicted)d

print('R-squared (training) ', round(regression.score(X_train, y_train), 3)) #0.549 #0.608

print('R-squared (testing) ', round(regression.score(X_test, y_test), 3)) #0.516 # 0.594

print('Intercept: ', regression.intercept_)

print('Coefficient:', regression.coef_)

