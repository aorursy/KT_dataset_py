# Code you have previously used to load data

import pandas as pd



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex3 import *



print("Setup Complete")
# print the list of columns in the dataset to find the name of the prediction target

home_data.columns
y = home_data['SalePrice']



# Check your answer

step_1.check()
# Create the list of features below

feature_names = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



# Select data corresponding to features in feature_names

X = home_data[feature_names]



# Check your answer

step_2.check()
# Review data

# print description or statistics from X

X.describe()
# print the top few lines

X.head()
from sklearn.tree import DecisionTreeRegressor

#specify the model. 

#For model reproducibility, set a numeric value for random_state when specifying the model

iowa_model = DecisionTreeRegressor(random_state=1)



# Fit the model

iowa_model.fit(X, y)



# Check your answer

step_3.check()
predictions = iowa_model.predict(X)

print(predictions)



# Check your answer

step_4.check()
# You can write code in this cell

predictions = pd.Series(predictions)

y_check = pd.concat([y, predictions], axis=1)

y_check.columns = (['y_true', 'y_pred'])

y_check['diff'] = y_check['y_true']-y_check['y_pred']



y_check.head(10)
# checking of there are any differences between predicted and true values in all of the dataset



y_check['diff'].value_counts()