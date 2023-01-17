# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
melbourne_file_path = '../input/melbourne-housing-snapshot/melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)

# print a summary of the data in Melbourne data
print(melbourne_data.describe())
print(melbourne_data.columns)
# store the series of prices separately as melbourne_price_data.
data_price = melbourne_data.Price
# the head command returns the top few lines of data.
print(melbourne_data.head())
columns_of_interest = ['Landsize', 'BuildingArea']
two_columns_of_data = melbourne_data[columns_of_interest]
two_columns_of_data.describe()
y = melbourne_data.Price
data_predictors = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = melbourne_data[data_predictors]
print(X.isnull().sum())

# Imputing values
from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(X)
type(data_with_imputed_values)
# Parsing to DataFrame
X = pd.DataFrame(data=data_with_imputed_values[0:,0:],    # values
             columns=data_predictors)  # 1st row as the column names
print(X.isnull().sum())

print(y.isnull().sum())
y = y.fillna(y.mean())
print(y.isnull().sum())
###################################
#### First model: Decision Tree ###
###################################
from sklearn.tree import DecisionTreeRegressor

# Define model
melbourne_model = DecisionTreeRegressor()

# Fit model
melbourne_model.fit(X, y)
print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
####################################
########## New Predictor ###########
####################################
new_melbourne_file_path = '../input/house-prices-advanced-regression-techniques/housetrain.csv'
new_melbourne_data = pd.read_csv(new_melbourne_file_path)

# print a summary of the data in Melbourne data
print(new_melbourne_data.describe())
new_y = new_melbourne_data.SalePrice
new_predictors = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]
new_X = new_melbourne_data[new_predictors]
print(X.isnull().sum())
print(y.isnull().sum())
###################################
#### First model: Decision Tree ###
###################################
from sklearn.tree import DecisionTreeRegressor

# Define model
new_melbourne_model = DecisionTreeRegressor()

# Fit model
new_melbourne_model.fit(new_X, new_y)
print("Making predictions for the following 5 houses:")
print(new_X.head())
print("The predictions are")
print(new_melbourne_model.predict(new_X.head()))
###################################
########### Validation ############
###################################
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
mean_absolute_error(y, predicted_home_prices)
# the second model has a better result
new_predicted_home_prices = new_melbourne_model.predict(new_X)
mean_absolute_error(new_y, new_predicted_home_prices)
from sklearn.model_selection import train_test_split

# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
train_new_X, val_new_X, train_new_y, val_new_y = train_test_split(new_X, new_y,random_state = 0)
# Define model
new_melbourne_model = DecisionTreeRegressor()
# Fit model
new_melbourne_model.fit(train_new_X, train_new_y)

# get predicted prices on validation data
new_val_predictions = new_melbourne_model.predict(val_new_X)
print(mean_absolute_error(val_new_y, new_val_predictions))
###########################################################
#### Underfitting, Overfitting and Model Optimization #####
###########################################################
from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

def get_mae(max_leaf_nodes, predictors_train, predictors_val, targ_train, targ_val):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(predictors_train, targ_train)
    preds_val = model.predict(predictors_val)
    mae = mean_absolute_error(targ_val, preds_val)
    return(mae)
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_new_X, val_new_X, train_new_y, val_new_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))
###########################################################
################# Random Forests ##########################
###########################################################
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
melb_preds = forest_model.predict(val_X)
print(mean_absolute_error(val_y, melb_preds))
forest_model.fit(train_new_X, train_new_y)
melb_new_preds = forest_model.predict(val_new_X)
print(mean_absolute_error(val_new_y, melb_new_preds))

