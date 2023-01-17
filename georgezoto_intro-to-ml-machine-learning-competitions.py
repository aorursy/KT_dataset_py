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

print('Data with missing values:{}', home_data.shape)



# Version 2 - Fill NAs + Use all numerical features

home_data = home_data.fillna(0)

home_data.dropna(inplace=True)

print('Data without missing values:{}', home_data.shape)



# Create target object and call it y

y = home_data.SalePrice

# Create X

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



# Version 2 - Fill NAs + Use all numerical features

features = []

for feature, data_type in home_data.dtypes.items():

    if (feature != 'SalePrice') & (data_type != 'object'):

        print(feature, data_type)

        features.append(feature)

        

print(features)        

X = home_data[features]
home_data.isna()
# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}, Tree Depth:{}".format(val_mae, iowa_model.get_depth()))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE   for best value of max_leaf_nodes: {:,.0f}, Tree Depth:{}".format(val_mae, iowa_model.get_depth()))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}, # of features:{}, # of outputs:{}".format(rf_val_mae, rf_model.n_features_, rf_model.n_outputs_))
home_data.columns
for feature, data_type in home_data.dtypes.items():

    if data_type != 'object':

        print(feature, data_type)
from matplotlib import pyplot as plt

from sklearn.inspection import permutation_importance



# the permutation based importance

perm_importance = permutation_importance(rf_model, val_X, val_y)



sorted_idx = perm_importance.importances_mean.argsort()

#print(features, sorted_idx)

#plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])

plt.figure(figsize=(32,18))

plt.xlabel("Permutation Importance")

plt.barh([x for _,x in sorted(zip(perm_importance.importances_mean, features))], perm_importance.importances_mean[sorted_idx]);
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)

test_data = test_data.fillna(0)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

print(features)

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# Check your answer

step_1.check()

#step_1.solution()