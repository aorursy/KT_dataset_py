# code you have previously used



# load data

import pandas as pd

iowa_file_path = '../input/home-data-for-ml-course/train.csv'

home_data = pd.read_csv(iowa_file_path)



# create target object and call it y

y = home_data['SalePrice']



# create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# split into validation and training data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# specify Model

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)



# fit Model

iowa_model.fit(train_X, train_y)



# make validation predictions

val_predictions = iowa_model.predict(val_X)



# calculate mean absolute error

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE when not specifying max_leaf_nodes: {val_mae:,.0f}")

# print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE for best value of max_leaf_nodes: {val_mae:,.0f}")



# set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex6 import *

print("\nSetup complete")
from sklearn.ensemble import RandomForestRegressor



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)



# fit your model

rf_model.fit(train_X, train_y)



# Calculate the mean absolute error of your Random Forest model on the validation data

val_ft_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(val_y, val_ft_predictions)



print(f"Validation MAE for Random Forest Model: {rf_val_mae}")



# Check your answer

step_1.check()
# The lines below will show you a hint or the solution.

# step_1.hint() 

# step_1.solution()