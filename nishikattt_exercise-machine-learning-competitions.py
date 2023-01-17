# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

print(home_data)
# To improve accuracy, create a new Random Forest model which you will train on all training data

def mae(hp_list, train_X, train_y, val_X, val_y):

    model = RandomForestRegressor(max_depth=hp_list[0], max_leaf_nodes=hp_list[1],random_state=1)

    model.fit(train_X, train_y)

    prediction = model.predict(val_X)

    mae = mean_absolute_error(prediction, val_y)

    return mae



max_depth_list = [2,4,6,8]

max_leaf_nodes = [2,4,8,16]

random_state=1

grid_search = []

for d in max_depth_list:

    for l in max_leaf_nodes:

        grid_search.append([d, l])



m = float('inf')



for hp_menu in grid_search:

        temp = mae(hp_menu, train_X, train_y, val_X, val_y)

        if m > temp:

            depth = hp_menu[0]

            leaf  = hp_menu[1]

            m = temp



print("best depth is", depth)

print("best leaf is",leaf)

model = RandomForestRegressor(max_depth=6, max_leaf_nodes=16,random_state=1)

model.fit(train_X, train_y)



# path to file you will use for predictions

test_data_path = '../input/test.csv'

#best depth is 6

#best leaf is 16

# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = model.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,                     'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# Check your answer

step_1.check()

# step_1.solution()