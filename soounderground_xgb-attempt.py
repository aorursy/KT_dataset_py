# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from catboost import CatBoostRegressor







# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)



print(type(home_data))



imputed_data = home_data.fillna(home_data.mean())



#imputing

#my_imputer = SimpleImputer()

#imputed_data = my_imputer.fit_transform(home_data)



# Create target object and call it y

y = imputed_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd',

            'YearRemodAdd', 'OverallQual', 'OverallCond', 'HeatingQC']

best_features_according_to_catboost = ['GrLivArea',

                                       'OverallQual',

                                       'GarageCars',

                                       'TotalBsmtSF',

                                       'BsmtQual',

                                       'Fireplaces',

                                       'ExterQual',

                                       '1stFlrSF',

                                       'YearRemodAdd',

                                       'KitchenQual',

                                       'LotArea',

                                       'GarageArea',

                                       'BsmtFinSF1',

                                       'BsmtFinType1',

                                       'TotRmsAbvGrd',

                                       'MSZoning',

                                       '2ndFlrSF',

                                       'FullBath',

                                       'SaleCondition',

                                       'OverallCond']

X = imputed_data[best_features_according_to_catboost]

print(imputed_data.columns)

#all_features_X = imputed_data.loc[:, imputed_data.columns != [['SalePrice', 'Exterior2nd']]]

all_features_X = imputed_data.drop(['SalePrice', 

                                    'Exterior2nd', 

                                    'RoofMatl',

                                    'Utilities',

                                    'Heating',

                                    'HouseStyle',

                                    'Exterior1st',

                                    'PoolQC',

                                    'RoofMatl',

                                    'Electrical',

                                    'Condition2',

                                    'GarageQual',

                                    'MiscFeature'], axis=1)

print(all_features_X)

#all_features_X = imputed_data - imputed_data.SalePrice

print(all_features_X.columns.values)



#s = pd.Series(list(['Ex', 'Gd', 'TA', 'Fa', 'Po']))

#pd.get_dummies('HeatingQC')

#print(pd.get_dummies(X))

X_dummies = pd.get_dummies(all_features_X)

#one-hot-encoded-X = pd.get_dummies(features)

all_X_dummies = pd.get_dummies(all_features_X)





# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(all_X_dummies, y, random_state=1)



# Make validation predictions and calculate mean absolute error

iowa_model = DecisionTreeRegressor(random_state=1)

iowa_model.fit(train_X, train_y) 

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



#XGB

xgb_model = XGBRegressor(n_estimators=1000)

xgb_model.fit(train_X, train_y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)

xgb_val_predictions = xgb_model.predict(val_X)

xgb_val_mae = mean_absolute_error(xgb_val_predictions, val_y)

print("Validation MAE for XGB Model: {:,.0f}".format(xgb_val_mae))



#CatBoost

#catboost_model = CatBoostRegressor()

#catboost_model.fit(train_X, train_y)

#print(catboost_model.get_feature_importance(data=None,

                      # prettified=True,

                      # thread_count=-1,

                      # verbose=False))

#catboost_model_val_predictions = catboost_model.predict(val_X)

#catboost_model_val_mae = mean_absolute_error(catboost_model_val_predictions, val_y)

#print("Validation MAE for Catboost Model: {:,.0f}".format(catboost_model_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data

xgb_model_on_full_data = XGBRegressor(n_estimators=1000)

#print(all_X_dummies.columns)

print(len(all_X_dummies.columns))

# fit rf_model_on_full_data on all data from the training data

xgb_model_on_full_data.fit(all_X_dummies, y, early_stopping_rounds=5, eval_set=[(val_X, val_y)], verbose=False)





# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



#imputing

my_imputer = SimpleImputer()

new_imputed_data  = test_data.fillna(home_data.mean())

#new_imputed_data = new_imputed_data.as_matrix()



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = new_imputed_data.drop(['Exterior2nd', 

                                    'RoofMatl',

                                    'Utilities',

                                    'Heating',

                                    'HouseStyle',

                                    'Exterior1st',

                                    'PoolQC',

                                    'RoofMatl',

                                    'Electrical',

                                    'Condition2',

                                    'GarageQual',

                                    'MiscFeature'], axis=1)

print(test_X.columns.values)

test_X_dummies = pd.get_dummies(test_X)

new_test_X_dummies = test_X_dummies

print(test_X_dummies.columns.values)

#test_X_dummies = pd.get_dummies(all_features_X)

#test_X = imputed_data[best_features_according_to_catboost]

#test_X_dummies = pd.get_dummies(all_features_X)



 # predict is not working without this code





# make predictions which we will submit. 



test_preds = xgb_model_on_full_data.predict(test_X_dummies)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)