# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from xgboost import XGBRegressor

from learntools.core import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)



home_data = home_data.drop(home_data['LotFrontage'][home_data['LotFrontage']>200].index)

home_data = home_data.drop(home_data['LotArea'][home_data['LotArea']>100000].index)

home_data = home_data.drop(home_data['BsmtFinSF1'][home_data['BsmtFinSF1']>4000].index)

home_data = home_data.drop(home_data['TotalBsmtSF'][home_data['TotalBsmtSF']>6000].index)

home_data = home_data.drop(home_data['1stFlrSF'][home_data['1stFlrSF']>4000].index)

home_data = home_data.drop(home_data.GrLivArea [(home_data['GrLivArea']>4000) & (home_data.SalePrice <300000)].index)

home_data = home_data.drop(home_data.LowQualFinSF [home_data['LowQualFinSF']>550].index)



# Create target object and call it y

y = home_data.SalePrice



# Create X

train_predictors = home_data.drop(['Id', 'SalePrice','GarageYrBlt', 'TotRmsAbvGrd', 'MiscVal', 'MiscFeature'], axis=1)



numeric_cols = [cname for cname in train_predictors.columns if train_predictors[cname].dtype in ['int64','float64']]

low_cardinality_cols = [cname for cname in train_predictors.columns if 

                                train_predictors[cname].nunique() < 30 and

                                train_predictors[cname].dtype == "object"]

features = numeric_cols + low_cardinality_cols



train_predictors = home_data[features].fillna(0)

X = pd.get_dummies(train_predictors)



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



# Define the model. Set random_state to 1

xgb_model = XGBRegressor(n_estimators=1400, learning_rate=0.04, min_child_weight=2)

xgb_model.fit(train_X, train_y)

xgb_val_predictions = xgb_model.predict(val_X)

rf_val_mae = mean_absolute_error(xgb_val_predictions, val_y)



print("Validation MAE for XGBoost Model: {:,.0f}".format(rf_val_mae))

# To improve accuracy, create a new Random Forest model which you will train on all training data

xgb_model_on_full_data = XGBRegressor(n_estimators=1400, learning_rate=0.04, min_child_weight=2)



#train_predictors_on_full_data = home_data[features]

#X = pd.get_dummies(train_predictors_on_full_data)



# fit rf_model_on_full_data on all data from the training data

xgb_model_on_full_data.fit(X,y)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_predictors = test_data[features]

test_X = pd.get_dummies(test_predictors)

final_train, final_test = X.align(test_X, join='left', axis=1)



# make predictions which we will submit. 

test_preds = xgb_model_on_full_data.predict(final_test)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)