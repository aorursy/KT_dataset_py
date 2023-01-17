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

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import LabelEncoder



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/train.csv'



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['BsmtExposure','Neighborhood','MSZoning','BldgType','LotConfig','PoolArea','LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd','GarageArea','GarageCars','KitchenAbvGr','GrLivArea','TotalBsmtSF','OverallQual','OverallCond','YearRemodAdd','GarageYrBlt','YrSold']

X = home_data[features]



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



# Specify Model

iowa_model = DecisionTreeRegressor(random_state=1)







enc = OneHotEncoder(sparse=False)

enco_X= pd.get_dummies(train_X)

encoval_X=pd.get_dummies(val_X)

enco1_X=pd.get_dummies(X)



my_imputer = SimpleImputer()

impu_X=my_imputer.fit_transform(enco1_X)

imputer_X = my_imputer.fit_transform(enco_X)

impuval_X=my_imputer.fit_transform(encoval_X)





# Fit Model

iowa_model.fit(imputer_X, train_y)



# Make validation predictions and calculate mean absolute error

val_predictions = iowa_model.predict(impuval_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes

iowa_model = DecisionTreeRegressor(max_leaf_nodes=700, random_state=1)

iowa_model.fit(imputer_X, train_y)

val_predictions = iowa_model.predict(impuval_X)

val_mae = mean_absolute_error(val_predictions, val_y)

print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



# Define the model. Set random_state to 1

rf_model = RandomForestRegressor(max_leaf_nodes=800,random_state=1)

rf_model.fit(imputer_X, train_y)

rf_val_predictions = rf_model.predict(impuval_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))
# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = RandomForestRegressor(max_leaf_nodes=800,random_state=1)



# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(impu_X,y)

pred=rf_model_on_full_data.predict(impu_X)

mae=mean_absolute_error(pred,y)

print("Validation MAE for Random Forest Model: {:,.0f}".format(mae))
from sklearn.impute import SimpleImputer



# path to file you will use for predictions

test_data_path = '../input/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]

encotest_X=pd.get_dummies(test_X)

my_imputer = SimpleImputer()

imp_X = my_imputer.fit_transform(encotest_X)



#imputes_y=my_imputer.transform(y)

#test_X.dtypes

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(imp_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

#rf_val_mae = mean_absolute_error(test_preds, y)

#print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

output = pd.DataFrame({'Id': test_data.Id,

                     'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)



#
step_1.check()

#step_1.solution()