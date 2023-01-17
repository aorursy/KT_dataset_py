#Step 1 : Import the packages required



#Package 1

#Package to read the csv file

import pandas as pd

import numpy as np



#Package 2

#Package to Scatter plot to see correlation

from matplotlib import pyplot as plt_training

from matplotlib import pyplot as plt_validation



#Package 3

#Required for linear regression

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score



#Package 4

#Determine SQRT 

import math
#Step 2 : Read the training and the validation data

#Read training data

train_path = r'../input/train.csv'

validation_path = r'../input/test.csv'



training_data = pd.read_csv(train_path)

validation_data = pd.read_csv(validation_path)
#Step 3 : Find and remove Na's from the dataset



#Find NA

print(len(training_data) - training_data.count())

print(len(training_data) - training_data.count())



#Drop NA in training dataset

training_data = training_data.dropna()
#Step 4: Scatter plot to determine the correlation

plt_training.scatter(training_data['x'], training_data['y'])

plt_training.title("Training dataset")

plt_training.xlabel("x train")

plt_training.ylabel("y train")
plt_validation.scatter(validation_data['x'], validation_data['y'])

plt_validation.title("validation dataset")

plt_validation.xlabel("x validation")

plt_validation.ylabel("y validation")
#Step 5 :Determining the linear model for training dataset

x_train = training_data[['x']]

y_train = training_data[['y']]

linear_model = LinearRegression()

linear_model.fit(x_train,y_train)

#Step 6 : Parameters

print("Coefficient:",round(linear_model.coef_[0][0],4))

print("Intercept:",round(linear_model.intercept_[0],4))

print("R^2 value :",linear_model.score(x_train,y_train))

print("Coefficient of Determination:", math.sqrt(linear_model.score(x_train,y_train)))
#Step 7 : Error coefficient

x_validation = validation_data[['x']]

y_validation = validation_data[['y']]



y_pred = linear_model.predict(x_validation)

error = y_validation - y_pred

print(error)
#Step 8 : Mean squared error

mean_squared_error(y_validation, y_pred)