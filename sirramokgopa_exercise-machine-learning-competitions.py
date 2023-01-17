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

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=2)



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


def get_mae(param, train_X, val_X, train_y, val_y):

    """A funtion to get the MAE"""

    model = RandomForestRegressor(n_estimators=param, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)







def lowest_mae(param_range,train_X, val_X, train_y, val_y):

    """A funtion to find the lowest MAE"""

    mae_holder = []  # Store the best value of max_leaf_nodes 

    for param in param_range:

        mae_holder.append(get_mae(param, train_X, val_X, train_y, val_y))

    best_param = param_range[mae_holder.index(min(mae_holder))]

    return best_param





# Define the model. 

n_estimators = range(60,80,1)

best_n_estimators = lowest_mae(n_estimators,train_X, val_X, train_y, val_y)



rf_model = RandomForestRegressor(n_estimators=best_n_estimators,random_state=1)

rf_model.fit(train_X, train_y)





# Test predictions predictions

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

print(f"The best number of estimators is: \t {best_n_estimators}")


list(estimator.tree_.max_depth for estimator in rf_model.estimators_)
def get_mae(param, train_X, val_X, train_y, val_y):

    """A funtion to get the MAE"""

    model = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=param, random_state=0)

    model.fit(train_X, train_y)

    preds_val = model.predict(val_X)

    mae = mean_absolute_error(val_y, preds_val)

    return(mae)





def lowest_mae(param_range,train_X, val_X, train_y, val_y):

    """A funtion to find the lowest MAE"""

    mae_holder = []  # Store the best value of max_leaf_nodes 

    for param in param_range:

        mae_holder.append(get_mae(param, train_X, val_X, train_y, val_y))

    best_param = param_range[mae_holder.index(min(mae_holder))]

    return best_param





# Define the model. 

depth = range(10,30,1)

best_max_depth = lowest_mae(depth,train_X, val_X, train_y, val_y)



rf_model = RandomForestRegressor(n_estimators=best_n_estimators, max_depth=best_n_estimators,random_state=1)

rf_model.fit(train_X, train_y)





# Test predictions predictions

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

print(f"The best max tree depth is: \t {best_max_depth}")
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(n_estimators=best_n_estimators,random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
step_1.check()

# step_1.solution()