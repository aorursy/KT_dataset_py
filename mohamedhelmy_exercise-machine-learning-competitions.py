# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *
# print(test_data.columns)

# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# test_data['GarageArea']
features1 = ['LotArea', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Remove NAN values in the whole data set

home_data1 = home_data

# Create target object and call it y

# y = home_data.SalePrice



y1 = home_data1.SalePrice



print(home_data1.dtypes)

print(home_data1.columns)
# Create X

# features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



features1 = ['LotArea', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']



# X = home_data[features]



X1 = home_data1[features1]



# Split into validation and training data

# train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



train_X1, val_X1, train_y1, val_y1 = train_test_split(X1, y1, random_state=1)
home_data.columns
# Comment out Decision Tree Regression for now

"""

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



"""
# Define the model. Set random_state to 1

"""

rf_model = RandomForestRegressor(random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model1.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

"""











rf_model1 = RandomForestRegressor(random_state=1)

rf_model1.fit(train_X1, train_y1)

rf_val_predictions1 = rf_model1.predict(val_X1)

rf_val_mae1 = mean_absolute_error(rf_val_predictions1, val_y1)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae1))

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(random_state = 1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X1, y1)

# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)
# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

#test_X = test_data[features]

test_X1 = test_data[features1]

print(test_X1.shape)

test_X1.dropna(inplace = True)

print(test_X1.shape)
test_X1.head(10)
# make predictions which we will submit. 

#test_preds = rf_model.predict(test_X)



test_preds1 = rf_model1.predict(test_X1)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds1})

#output.to_csv('submission1.csv', index=False)





output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds1})

output.to_csv('submission1.csv', index=False)
# Add LotShape, Condition1 and Neighborhood to features that will be used in building the model (regression)

# Condition 1 does not improve regression, add BldgType to features

# BldgType improves accuracy, use OverallQual

# OverallQual improves accuracy significantly, use OverallCond

# OverallCond improved accuracy, use Exterior Quality

# Exterior Quality did not improve accuracy, use exterior condition, ExterCond

# ExterCond did not improve quality, what if both are used

# Use Basement Exposure

# Use PavedDrive

features2_Sale = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea', 

                  'LotShape', 'Neighborhood', 'BldgType', 'OverallQual', 'OverallCond', 'ExterCond', 'ExterQual', 

                  'BsmtExposure', 'PavedDrive', 'TotalBsmtSF', 'SalePrice']



# work on more features

home_data2 = home_data[features2_Sale]

home_data2.dropna(inplace = True)



home_data2.shape
features2 = features2_Sale.copy()

features2.remove('SalePrice')

print(features2)
#features2 =  ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd', 'GrLivArea', 'LotShape', 'Neighborhood']



X2 = pd.get_dummies(home_data2[features2], drop_first = True)

y2 = home_data2['SalePrice']



print(X2.shape)

print(X2.columns)



train_X2, val_X2, train_y2, val_y2 = train_test_split(X2, y2, random_state = 1)



rf_model2 = RandomForestRegressor(random_state = 1)



rf_model2.fit(train_X2, train_y2)



rf_val_predictions2 = rf_model2.predict(val_X2)

rf_val_mae2 = mean_absolute_error(rf_val_predictions2, val_y2)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae2))
# Test Data

test_X2 = pd.get_dummies(test_data[features2], drop_first = True)

print(test_X2.shape)

print(test_X2.columns)
test_X2.fillna(test_X2.mean(), inplace = True)



test_preds2 = rf_model2.predict(test_X2)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



#output = pd.DataFrame({'Id': test_data.Id,

#                       'SalePrice': test_preds1})

#output.to_csv('submission1.csv', index=False)





output = pd.DataFrame({ 'Id': test_data.Id,

                       'SalePrice': test_preds2})

output.to_csv('submission2.csv', index=False)


