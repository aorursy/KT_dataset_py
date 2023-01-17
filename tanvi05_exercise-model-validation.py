# Code you have previously used to load data

import pandas as pd

from sklearn.tree import DecisionTreeRegressor



# Path of the file to read

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



home_data = pd.read_csv(iowa_file_path)

y = home_data.SalePrice

feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[feature_columns]



# Specify Model

iowa_model = DecisionTreeRegressor()

# Fit Model

iowa_model.fit(X, y)



print("First in-sample predictions:", iowa_model.predict(X.head()))

print("Actual target values for those homes:", y.head().tolist())



# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex4 import *

print("Setup Complete")
# Import the train_test_split function and uncomment

from sklearn.model_selection import train_test_split



train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 1)



# Check your answer

step_1.check()
iowa_model = DecisionTreeRegressor(random_state = 1)



# Fit iowa_model with the training data.

iowa_model.fit(train_X, train_y)



# Check your answer

step_2.check()
# Predict with all validation observations

val_predictions = iowa_model.predict(val_X)



# Check your answer

step_3.check()
print(val_y.head())

print(val_predictions)
from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)



# uncomment following line to see the validation_mae

print(val_mae)



# Check your answer

step_4.check()