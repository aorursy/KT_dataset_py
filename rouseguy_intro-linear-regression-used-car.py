#import the required libraries

import pandas as pd

import numpy as np
#Read the dataset

cars = pd.read_csv("../input/autos.csv", encoding='latin1')
#Display the first few rows

cars.head()
#Display the columns in the dataset

cars.columns
#what are the types of the columns?

cars.dtypes
#Find if data has missing values?

#Find missing values by each column

cars.isnull().sum()
#Find proportion of data that is missing for each of the columns

cars.isnull().sum()/cars.shape[0] * 100
#For this exercise, let's drop the rows that have null values



cars_updated = cars.dropna()
cars.shape, cars_updated.shape
#check if there are any missing values

cars_updated.isnull().sum()
#Display first few records of cars_updated

cars_updated.head()
cars_updated.columns
#Let's use only the following columns for our modeling now

cars_updated = cars_updated.iloc[:, [2,3,6,7,8,9,10,11,12,13,14,15,4]]
#Convert text to numeric using Label Encoding

from sklearn import preprocessing
#encode the data

cars_encoded = cars_updated.apply(preprocessing.LabelEncoder().fit_transform)
#Display the first few records

cars_encoded.head()
cars_encoded.columns
#Exploratory data analysis

import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('fivethirtyeight')

# plt.rcParams['figure.figsize'] = (10, 6)
#Plot year vs price

plt.scatter(cars_encoded.yearOfRegistration, 

           cars_encoded.price,

           s=150, alpha = 0.1)

plt.xlabel('year')

plt.ylabel('price')
from sklearn import linear_model
#Instantiate the model

model_sklearn = linear_model.LinearRegression()
#fit the model

model_sklearn.fit(cars_encoded.iloc[:,:12], cars_encoded.iloc[:,12])
#Regression coefficients

model_sklearn.coef_
#Model intercept

model_sklearn.intercept_
from sklearn.model_selection import train_test_split
#Split into train and validation

x_train, x_test, y_train, y_test = train_test_split(cars_encoded.iloc[:,:12], 

                                                    cars_encoded.iloc[:,12],

                                                    test_size=0.2)
#Display data shape

cars_encoded.shape, x_train.shape, y_train.shape, x_test.shape, y_test.shape
#Instantiate the model

model_sklearn_tv = linear_model.LinearRegression()
#fit the model

model_sklearn_tv.fit(x_train, y_train)
y_pred = model_sklearn_tv.predict(x_test)
#Find error : RMSE

np.sqrt(np.mean((y_test - y_pred)**2))