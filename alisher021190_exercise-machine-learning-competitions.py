# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

#print(home_data.dtypes)

# Create target object and call it y

y = home_data.SalePrice

# Create X

#home_data2=home_data.select_dtypes(exclude=['object'])

#print(home_data2.columns)

features = ['MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual',

       'OverallCond', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1',

       'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF',

       'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea',

       'MiscVal', 'MoSold', 'YrSold']



X = home_data[features]

cols_with_missing = [col for col in X.columns 

                                 if X[col].isnull().any()]

X = X.drop(cols_with_missing, axis=1)

# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

cols_with_missing = [col for col in train_X.columns 

                                 if train_X[col].isnull().any()]

reduced_X_train = train_X.drop(cols_with_missing, axis=1)

reduced_X_test  = test_X.drop(cols_with_missing, axis=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(reduced_X_train, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(reduced_X_train, train_y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(reduced_X_train, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=2)

features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',

       'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',

       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



X = home_data[features]

cols_with_missing = [col for col in X.columns 

                                 if X[col].isnull().any()]

X = X.drop(cols_with_missing, axis=1)



rf_model_on_full_data.fit(X,y)

predictions_full = rf_model_on_full_data.predict(X)

mae_full=mean_absolute_error(predictions_full,y)

print(mae_full)

# fit rf_model_on_full_data on all data from the training data

____

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



features = ['MSSubClass', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt',

       'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea',

       'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',

       'Fireplaces', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch',

       'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']



X = test_data[features]

cols_with_missing = [col for col in X.columns 

                                 if X[col].isnull().any()]

test_X = X.drop(cols_with_missing, axis=1)

print(test_X.columns)

# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features



#test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

#print(test_preds)

output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)