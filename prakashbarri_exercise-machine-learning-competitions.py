# Code you have previously used to load data

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.experimental import enable_hist_gradient_boosting

from xgboost import XGBRegressor

from sklearn.ensemble import ExtraTreesRegressor

from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeRegressor

from learntools.core import *

import os

import warnings

warnings.filterwarnings("ignore")





def getmea(numnodes,train_X,train_y,val_X,val_y):

    rf_model = GradientBoostingRegressor(n_estimators=numnodes,random_state=1)

    rf_model.fit(train_X, train_y)

    rf_val_predictions = rf_model.predict(val_X)

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    return rf_val_mae



def getmea1(numnodes,X,y):

    #rf_model = ExtraTreesRegressor(n_estimators=numnodes,random_state=1)

    rf_model=XGBRegressor(n_estimators=numnodes, learning_rate=0.2, n_jobs=10)

    from sklearn.model_selection import cross_val_score

    import numpy as np

    scores = cross_val_score(rf_model,X,y,cv=10,scoring="neg_mean_absolute_error")

    print(scores)

    return scores.mean()



def getmea2(numnodes,X,y):

    #rf_model = ExtraTreesRegressor(n_estimators=numnodes,random_state=1)

    rf_model=RandomForestRegressor(n_estimators=numnodes,n_jobs=-1)

    from sklearn.model_selection import cross_val_score

    import numpy as np

    scores = cross_val_score(rf_model,X,y,cv=10,scoring="neg_mean_absolute_error")

    print(scores)

    return scores.mean()



# Path of the file to read. We changed the directory structure to simplify submitting to a competition

iowa_file_path = '../input/home-data-for-ml-course/train.csv'



# print(os.listdir("../input/home-data-for-ml-course/"))



home_data = pd.read_csv(iowa_file_path)

# Create target object and call it y

y = home_data.SalePrice

# Create X

features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']

features = ['OverallQual','GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','YearBuilt','YearRemodAdd','GarageYrBlt','MasVnrArea','Fireplaces','BsmtFinSF1','LotFrontage','WoodDeckSF','2ndFlrSF','OpenPorchSF','HalfBath','LotArea','BsmtFullBath','BsmtUnfSF']

X = home_data[features]

# X = X.dropna(axis=1)

X = X.fillna(0)



# Split into validation and training data

train_X, val_X, train_y, val_y = train_test_split(X, y,test_size=0.3, random_state=1)



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

rf_model = RandomForestRegressor(n_estimators=700,random_state=1)

rf_model.fit(train_X, train_y)

rf_val_predictions = rf_model.predict(val_X)

rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)



print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



for i in range(100,100,100):

#     xyz = getmea(i,train_X,train_y,val_X,val_y)

#     print("Validation MAE for Random Forest Model:" +str(i)+" Nodes {:,.0f}".format(xyz))

    xyz = getmea2(i,X,y)

    print("Validation MAE for Random Forest Model:" +str(i)+" Nodes " + str(xyz))

#     pass



# 23071

# To improve accuracy, create a new Random Forest model which you will train on all training data

rf_model_on_full_data = XGBRegressor(n_estimators=150, learning_rate=0.05,n_jobs=-1)

# rf_model_on_full_data = RandomForestRegressor(n_estimators=80,n_jobs=-1)

# print(help(XGBRegressor))

# fit rf_model_on_full_data on all data from the training data

rf_model_on_full_data.fit(X,y)

# path to file you will use for predictions

test_data_path = '../input/home-data-for-ml-course/test.csv'



# read test data file using pandas

test_data = pd.read_csv(test_data_path)



# print(test_data.columns)



# create test_X which comes from test_data but includes only the columns you used for prediction.

# The list of columns is stored in a variable called features

test_X = test_data[features]

test_X = test_X.fillna(0)

# make predictions which we will submit. 

test_preds = rf_model_on_full_data.predict(test_X)



# The lines below shows how to save predictions in format used for competition scoring

# Just uncomment them.

#!pip install matplotlib



output = pd.DataFrame({'Id': test_data.Id,

                      'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)