# Code you have previously used to load data

import pandas as pd

import numpy as np

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *
# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

home_data.columns
def feature_engineering(dataframe):

    utility_mapping = {'AllPub': 2, 'NoSeWa': 1, np.nan: 0}

    dataframe['Utilities'] = dataframe['Utilities'].map(utility_mapping).astype(int)

    dataframe.loc[ dataframe['TotalBsmtSF'] >= 6000, 'TotalBsmtSF' ] = 5

    dataframe.loc[ (dataframe['TotalBsmtSF'] < 6000) & (dataframe['TotalBsmtSF'] >= 1700), 'TotalBsmtSF' ] = 4

    dataframe.loc[ (dataframe['TotalBsmtSF'] < 1700) & (dataframe['TotalBsmtSF'] >= 900), 'TotalBsmtSF' ] = 3

    dataframe.loc[ (dataframe['TotalBsmtSF'] < 900) & (dataframe['TotalBsmtSF'] >= 500), 'TotalBsmtSF' ] = 2

    dataframe.loc[ dataframe['TotalBsmtSF'] < 500, 'TotalBsmtSF' ] = 1

    dataframe.loc[ dataframe['TotalBsmtSF'].isna(), 'TotalBsmtSF' ] = 0

    dataframe['Foundation'] = dataframe['Foundation'].map({'BrkTil': 1, 'CBlock': 2, 'PConc': 3, 'Slab': 4, 'Stone': 5, 'Wood': 6}).astype(int)



feature_engineering(home_data)
# Target Y

Y = home_data.SalePrice



# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', \

            '2ndFlrSF', 'FullBath', 'BedroomAbvGr', \

            'TotRmsAbvGrd', 'Utilities', 'OverallQual', 

            'OverallCond', 'TotalBsmtSF', \

            'Foundation']

X = home_data[features]

X.head()


# Split into validation and training data

train_X, val_X, train_Y, val_Y = train_test_split(X, Y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)

# Fit Model

iowa_model.fit(train_X, train_Y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_Y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)

iowa_model.fit(train_X, train_Y)

val_predictions = iowa_model.predict(val_X)

val_mae = mean_absolute_error(val_predictions, val_Y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_Y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_Y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

# 

def get_mae(train_X, val_X, train_Y, val_Y, num_estimator=10, max_features="auto"):

    rf_model = RandomForestRegressor(n_estimators=num_estimator, random_state=0, max_features=max_features)

    rf_model.fit(train_X, train_Y)

    preds_val = rf_model.predict(val_X)

    mae = mean_absolute_error(val_Y, preds_val)

    return(mae)



number_of_estimators = [1, 5, 10, 20, 50, 70, 100]

estimators_mae_scores = {num_estimator: get_mae(train_X, val_X, train_Y, val_Y, num_estimator=num_estimator) for num_estimator in number_of_estimators} 

best_num_estimator = min(estimators_mae_scores, key=estimators_mae_scores.get)

print(estimators_mae_scores)

print("Best number of estimator to use:", best_num_estimator)



max_features = ["auto", "sqrt", "log2"]

max_features_mae_scores = {max_feature: get_mae(train_X, val_X, train_Y, val_Y, max_features=max_feature) for max_feature in max_features} 

best_max_feature = min(max_features_mae_scores, key=max_features_mae_scores.get)

print(max_features_mae_scores)

print("Best number of estimator to use:", best_max_feature)



optimized_rf_model = RandomForestRegressor(n_estimators=best_num_estimator, max_features=best_max_feature, random_state=1)

optimized_rf_model.fit(train_X, train_Y)

optimized_rf_val_predictions = optimized_rf_model.predict(val_X)

optimized_rf_val_mae = mean_absolute_error(optimized_rf_val_predictions, val_Y)



print("Optimzed mae score: ", optimized_rf_val_mae)
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(n_estimators=best_num_estimator, max_features=best_max_feature, random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_Y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# feature engineering

feature_engineering(test_data)



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