# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor



# Set up code checking

import os

if not os.path.exists("../input/train.csv"):

    os.symlink("../input/home-data-for-ml-course/train.csv", "../input/train.csv")  

    os.symlink("../input/home-data-for-ml-course/test.csv", "../input/test.csv") 

from learntools.core import binder

binder.bind(globals())

from learntools.machine_learning.ex7 import *



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)



# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC', 'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType', 'SaleCondition']

categ_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities',

       'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

       'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',

       'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation',

       'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',

       'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',

       'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual',

       'GarageCond', 'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature',

       'SaleType', 'SaleCondition']

num_features=[x for x in features if x not in categ_features]





from sklearn import preprocessing

from sklearn.impute import SimpleImputer

import numpy as np



#Imputing missing values before

#missing for numbers use mean value

imp_nan_const = SimpleImputer(missing_values=np.nan, strategy='most_frequent', fill_value=0)

imp_nan_const.fit(home_data[num_features])

step00_X=pd.DataFrame(imp_nan_const.transform(home_data[num_features]))

step00_X.columns=home_data[num_features].columns

#for missing str use NA value

imp_nan_const1 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NA')

imp_nan_const1.fit(home_data[categ_features])

step01_X=pd.DataFrame(imp_nan_const1.transform(home_data[categ_features]))

step01_X.columns=home_data[categ_features].columns



##encode categorical features to numbers

le = preprocessing.LabelEncoder()

step10_X = step01_X.apply(le.fit_transform)

step10_X.head()



###contat categorican and numbers

step20_X=pd.concat([step10_X,step00_X], axis=1)

step20_X.head()



X = step20_X



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

iowa_model = DecisionTreeRegressor(max_leaf_nodes=1000, random_state=1)

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

rf_model_on_full_data = DecisionTreeRegressor(max_leaf_nodes=1000, random_state=1)

rf_model_on_full_data1 = RandomForestRegressor(random_state=1)





# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X, y)

rf_model_on_full_data1.fit(X, y)
# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



###----------------------------

# Create X

#Imputing missing values before



#missing for numbers use mean value

#imp_nan_const = SimpleImputer(missing_values=np.nan)

#imp_nan_const.fit(test_data[num_features])

t_step00_X=pd.DataFrame(imp_nan_const.transform(test_data[num_features]))

t_step00_X.columns=test_data[num_features].columns



#for missing str use NA value

#imp_nan_const1 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value='NA')

#imp_nan_const1.fit(test_data[categ_features])

t_step01_X=pd.DataFrame(imp_nan_const1.transform(test_data[categ_features]))

t_step01_X.columns=test_data[categ_features].columns



##encode categorical features to numbers

#le = preprocessing.LabelEncoder()

t_step10_X = t_step01_X.apply(le.fit_transform)

t_step10_X.head()



###contat categorican and numbers

t_step20_X=pd.concat([t_step10_X,t_step00_X], axis=1)

t_step20_X.head()



X = t_step20_X

###----------------------------



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = X #test_data[features]



# make predictions which we will submit. 

test_preds = rf_model_on_full_data1.predict(test_X)





# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.



output = pd.DataFrame({'Id': test_data.Id,

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)
step_1.check()

# step_1.solution()