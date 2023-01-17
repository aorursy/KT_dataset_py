# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

from sklearn.tree import DecisionTreeRegressor

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np

import seaborn as sns

from sklearn.utils import shuffle

from operator import itemgetter



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



pd.set_option('display.max_rows', 50)



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

# Load the data from the path

home_data = pd.read_csv(iowa_file_path)
# find any columns that have numeric values

features_new = list(home_data.select_dtypes(include=[np.number]).columns.values)

features_new.remove('SalePrice')



# Create target object and call it y

y = home_data.SalePrice



# Create X: Use every numerical columns

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

X = home_data[features_new]



# Check if there's any NaN value -> LotFrontage & GarageYrBlt

print(X.isnull().sum())



# Replace NaN with the column's mean

X = X.fillna(X.mean())





# X and y needs to be shuffled

#X, y = shuffle(X, y, random_state=1)



# Normalization/Standardization

scaler = MinMaxScaler()

#scaler = StandardScaler()

scaled_X = scaler.fit_transform(X)



# Check if y is skewed

print(y.skew())  # Series

sns.distplot(y)

log_y = np.log(y)  # Series

#sns.distplot(log_y)
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(scaled_X, log_y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

# Unlog by np.e**

val_predictions = np.e**iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = np.e**iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(n_estimators=100, random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = np.e**rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# Set the parameters by cross-validation

tuned_parameters = {'n_estimators': (10,50,100), 'max_depth': range(2,8),

                     'max_leaf_nodes': (10,50,100)}



#tuned_parameters = {'n_estimators': (10,50,100), 'max_depth': range(2,7),

#                     'max_leaf_nodes': (10,50,100)}



# Create a CV model (gscv = GridSearchCV)

rf_model_on_full_data = GridSearchCV(estimator=RandomForestRegressor(), param_grid = tuned_parameters, scoring = 'neg_mean_absolute_error', cv=5)

rf_model_on_full_data.fit(scaled_X, log_y)



"""

# Divide attributes in dic

n_estimators, max_depth, max_leaf_nodes = itemgetter('n_estimators', 'max_depth', 'max_leaf_nodes')(gscv.best_params_)



rf_model = RandomForestRegressor(n_estimators=n_estimators, max_depth = max_depth, max_leaf_nodes=max_leaf_nodes, random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = np.e**rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

print(mean_absolute_error(np.e**gscv.predict(val_X), val_y))



# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data -> Wrong validation sets cannot be used again

rf_model_on_full_data.fit(train_X, train_y)

"""
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features_new]



# Check if there's NAN

print(test_X.isnull().sum())



# Replace NaN with the column's mean

test_X = test_X.fillna(test_X.mean())



# StandardScaler

scaled_test_X = scaler.fit_transform(test_X)
# make predictions which we will submit. 

test_preds = np.e**rf_model_on_full_data.predict(scaled_test_X)



# Check if there is label -> No such label -> cannot calculate MAE

#print([l for l in test_data.columns if 'Price' in l])



# Create target object and call it test_y, and estimate its MAE

#test_y = test_data.SalePrice

#test_mae = mean_absolute_error(test_preds, test_y)

#print("Estimation MAE for Random Forest Model: {:,.0f}".format(test_mae))



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)



print("Done")
step_1.check()

# step_1.solution()