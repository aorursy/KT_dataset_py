# Code you have previously used to load data

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from sklearn.impute import SimpleImputer

from learntools.core import *







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# # Create target object and call it y

# y = home_data.SalePrice

# # Create X

# features = [

#     'LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 

#     'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd'

# ]

# X = home_data[features]



# # Split into validation and training data

# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# # Specify Model

# iowa_model = DecisionTreeRegressor(random_state=1)

# # Fit Model

# iowa_model.fit(train_X, train_y)



# # Make validation predictions and calculate mean absolute error

# val_predictions = iowa_model.predict(val_X)

# val_mae = mean_absolute_error(val_predictions, val_y)

# print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# # Using best value for max_leaf_nodes

# iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

# iowa_model.fit(train_X, train_y)

# val_predictions = iowa_model.predict(val_X)

# val_mae = mean_absolute_error(val_predictions, val_y)

# print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# # Define the model. Set random_state to 1

# rf_model = RandomForestRegressor(random_state=1)

# rf_model.fit(train_X, train_y)

# rf_val_predictions = rf_model.predict(val_X)

# rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



# print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



# Imputation Practice



def get_error(X_train, X_test, y_train, y_test):

    _model = RandomForestRegressor(random_state=1)

    _model.fit(X_train, y_train)

    _preds = _model.predict(X_test)

    _mae = mean_absolute_error(y_test, _preds)

    return _mae



iowa_target = home_data.SalePrice

iowa_predictors = home_data.drop(['SalePrice'], axis=1)



# Take only numeric data

iowa_numeric_predictors = iowa_predictors.select_dtypes(include=[np.number])

# print(iowa_numeric_predictors)



X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors, 

                                                    iowa_target, 

                                                    random_state=1)



# Dropping columns with NA

cols_with_miss = [col for col in X_train.columns if X_train[col].isnull().any()]

# print(cols_with_miss)

reduced_X_train = X_train.drop(cols_with_miss, axis=1)

reduced_X_test = X_test.drop(cols_with_miss, axis=1)

print(f'Validation MAE with Dropping columns with NA: {get_error(reduced_X_train, reduced_X_test, y_train, y_test)}')



# Imputation

my_test_imputer = SimpleImputer()

imputed_X_train = my_test_imputer.fit_transform(X_train)

imputed_X_test = my_test_imputer.fit_transform(X_test)

print(f'Validation MAE with Imputation: {get_error(imputed_X_train, imputed_X_test, y_train, y_test)}')



# Imputation while tracking what columns were imputed

imputed_X_train_plus = X_train.copy()

imputed_X_test_plus = X_test.copy()



for col in cols_with_miss:

    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()

    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()



imputed_X_train_plus = my_test_imputer.fit_transform(imputed_X_train_plus)

imputed_X_test_plus = my_test_imputer.fit_transform(imputed_X_test_plus)



print(f'Validation MAE with Imputation while tracking what was imputed: {get_error(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test)}')



X = iowa_numeric_predictors

y = home_data.SalePrice

reduced_X = X.drop(cols_with_miss, axis=1)

# print(reduced_X.columns)



my_imputer = SimpleImputer()

# imputed_X = my_imputer.fit_transform(X)

# # print(imputed_X)



imputed_X_plus = X.copy()

for col in cols_with_miss:

    imputed_X_plus[col + '_was_missing'] = imputed_X_plus[col].isnull()

imputed_X_plus = my_imputer.fit_transform(imputed_X_plus)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

# rf_model_on_full_data.fit(imputed_X, y)

rf_model_on_full_data.fit(imputed_X_plus, y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[iowa_numeric_predictors.columns]



# imputed_test_X = my_imputer.fit_transform(test_X)



imputed_test_X_plus = test_X.copy()

for col in cols_with_miss:

    imputed_test_X_plus[col + '_was_missing'] = imputed_test_X_plus[col].isnull()

imputed_test_X_plus = my_imputer.fit_transform(imputed_test_X_plus)



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(imputed_test_X_plus)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)