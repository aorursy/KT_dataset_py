# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from learntools.core import *







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

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor()



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)



# My remark - this model uses base ('0') set of 7 features:

# 'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'
# Feature set 'E12' : attempting 12 quantitative variables

# MSSubClass - The building class

# LotFrontage - Linear feet of street connected to property

# OverallQual - Overall material and finish quality

# OverallCond - Overall condition rating

# YearBuilt - Original construction date

# YearRemodAdd - Remodel date

# TotalBsmtSF - Total square feet of basement area

# 1stFlrSF - First Floor square feet

# GrLivArea - Above grade (ground) living area square feet

# FullBath - Full bathrooms above grade

# TotRmsAbvGrd - Total rooms above grade (does not include bathrooms)

# GarageCars - Size of garage in car capacity



home_data = pd.read_csv(iowa_file_path)



y = home_data.SalePrice



features = ['MSSubClass', 'LotFrontage', 'OverallQual', 'OverallCond',

            'YearBuilt', 'YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', 'GrLivArea',

            'FullBath', 'TotRmsAbvGrd', 'GarageCars']



X = home_data[features]

X['LotFrontage'].fillna(X['LotFrontage'].mean(), inplace=True)



# scaling train data

#scaler = StandardScaler(copy=True, with_mean=True, with_std=True)

#X = scaler.fit_transform(X)





rf_model_on_full_data = RandomForestRegressor()



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features



test_X = test_data[features]

test_X['LotFrontage'].fillna(test_X['LotFrontage'].mean(), inplace=True)

test_X['TotalBsmtSF'].fillna(test_X['TotalBsmtSF'].mean(), inplace=True)

test_X['GarageCars'].fillna(test_X['GarageCars'].mean(), inplace=True)



# scaling test data

#test_X = scaler.transform(test_X)



# make predictions which we will submit. 



test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)