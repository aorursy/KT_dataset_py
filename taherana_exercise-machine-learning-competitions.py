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

rf_model_on_full_data = DecisionTreeRegressor(random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(train_X, train_y)

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error



# path to file you will use for predictions

#test_data_path = '../input/test.csv'



# read test data file using pandas

#test_data = pd.read_csv(test_data_path)

#print(test_data.describe())

#print(test_data.head())

#print(test_data.columns)

# handle missing

#print('# of missings')

#print(test_data.columns)

#print(test_data.isna().sum())



#test_data = test_data.dropna(axis=0)

#print(test_data.describe())

#print(test_data.info())



#obj_col = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 

   ##        'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood','Condition1', 

    #       'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 

    #       'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 

     #      'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

    #       'BsmtFinType2','Heating', 'HeatingQC', 'CentralAir', 'Electrical', 

    #       'KitchenQual', 'Functional', 'FireplaceQu','GarageType', 'GarageFinish',

     #      'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC','Fence', 

    #       'MiscFeature', 'SaleType', 'SaleCondition', 

    #       'GarageArea','GarageCars','GarageYrBlt','BsmtHalfBath','BsmtFullBath',

      #     'TotalBsmtSF','BsmtUnfSF','BsmtFinSF2','BsmtFinSF1','MasVnrArea',

     #      'LotFrontage',

    #       ]

# remove obj attributes

#test_data = test_data.drop(obj_col, axis=1)

#print(test_data.info())

#y = test_data['Id']

#X = test_data.drop(['Id'], axis=1)

#print('y', y.head())

#print('X columns', X.columns)

# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

# Create target object and call it y

#y = home_data.SalePrice

# Create X

#features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

#X = home_data[features]

#X = home_data[features]

# make predictions which we will submit. 

#test_preds = ____

#for i in range(100, 301, 100):

#  results[i] = result



#best_est = sorted([(v, k) for k, v in results.items()])[0][1]

#print('best estimator num', best_est)



#print('evaluation ')

#rf_model_on_full_data = RandomForestRegressor(n_estimators=best_est)

#rf_model_on_full_data.fit(X_train, y_train)

#test_preds = rf_model_on_full_data.predict(test_X)

#result = mean_absolute_error(y_true=y_test, y_pred=test_preds)

#print('result ', result)

#test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



#output = pd.DataFrame({'Id': test_data.Id,

        #             'SalePrice': test_preds})

#output.to_csv('submission.csv', index=False)



# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

test_X = test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)