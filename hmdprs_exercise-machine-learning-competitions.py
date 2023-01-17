# set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *
# load train data

import pandas as pd

# path of the file to read.

# the directory structure was changed to simplify submitting to a competition

iowa_file_path = '../input/train.csv'

home_data = pd.read_csv(iowa_file_path)
# create target object and call it y

y = home_data['SalePrice']

# create X

features_numeric_nona = [

    "MSSubClass",  # The building class [20-190]

    "LotArea",  # Lot size in square feet [1300-215000]

    "1stFlrSF",  # First Floor square feet [334-4690]

    "2ndFlrSF",  # Second floor square feet [0-2690]

    "LowQualFinSF",  # Low quality finished square feet (all floors) [0-572]

    "GrLivArea",  # Above grade (ground) living area square feet [334-5642]

    "WoodDeckSF",  # Wood deck area in square feet [0-857]

    "OpenPorchSF",  # Open porch area in square feet [0-547]

    "EnclosedPorch",  # Enclosed porch area in square feet [0-552]

    "3SsnPorch",  # Three season porch area in square feet [0-508]

    "ScreenPorch",  # Screen porch area in square feet [0-480]

    "PoolArea",  # Pool area in square feet [0-738]

    "MiscVal",  # $Value of miscellaneous feature [0-15500]

]

features_numeric_na = [

    "BsmtFinSF1",  # Type 1 finished square feet [0-5644]

    "BsmtFinSF2",  # Type 2 finished square feet [0-1474]

    "BsmtUnfSF",  # Unfinished square feet of basement area [0-2336]

    "TotalBsmtSF",  # Total square feet of basement area [0-6110]

    "GarageArea",  # Size of garage in square feet [0-1418]

]

features_numeric = features_numeric_nona + features_numeric_na  # mae = 9,481

features_date = [

    "YearBuilt",  # Original construction date [1872-2010]

    "YearRemodAdd",  # Remodel date [1950-2010]

    "MoSold",  # Month Sold [1-12]

    "YrSold",  # Year Sold [2006-2010]

]  # mae = 22,972

features = features_numeric_nona + features_date  # mae = 8,764

X = home_data[features]
# split into validation and training data

from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
# specify model

from sklearn.tree import DecisionTreeRegressor

iowa_model = DecisionTreeRegressor(random_state=1)

# fit model

iowa_model.fit(train_X, train_y)

# make validation predictions

val_predictions = iowa_model.predict(val_X)

# calculate mae

from sklearn.metrics import mean_absolute_error

val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE when not specifying max_leaf_nodes: {val_mae:,.0f}")
# using best value for max_leaf_nodes

# specify model

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

# fit model

iowa_model.fit(train_X, train_y)

# make validation predictions

val_predictions = iowa_model.predict(val_X)

# calculate mae

val_mae = mean_absolute_error(val_y, val_predictions)

print(f"Validation MAE for best value of max_leaf_nodes: {val_mae:,.0f}")
# specify model

from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(random_state=1)

# fit model

rf_model.fit(train_X, train_y)

# make validation predictions

rf_val_predictions = rf_model.predict(val_X)

# calculate mae

rf_val_mae = mean_absolute_error(val_y, rf_val_predictions)

print(f"Validation MAE for Random Forest Model: {rf_val_mae:,.0f}")
# to improve accuracy, create a new Random Forest model which you will train on all training data

# specify model

rf_model_on_full_data = RandomForestRegressor(random_state=1)

# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

# make full data predictions

rf_model_on_full_data_predictions = rf_model_on_full_data.predict(X)

# calculate mae

rf_model_on_full_data_mae = mean_absolute_error(y, rf_model_on_full_data_predictions)

print(f"Full Data MAE for Random Forest Model: {rf_model_on_full_data_mae:,.0f}")
# load test data

# path to file you will use for predictions

test_data_path = '../input/test.csv'

# read test data file using pandas

test_data = pd.read_csv(test_data_path)
# create test_X which comes from test_data but includes only the columns you used for prediction.

# the list of columns is stored in a variable called features

test_X = test_data[features]

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)
# save predictions in format used for competition scoring

output = pd.DataFrame({'Id': test_data.Id, 'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
# Check your answer

step_1.check()

# step_1.solution()