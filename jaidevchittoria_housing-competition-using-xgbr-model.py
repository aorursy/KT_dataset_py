
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor, XGBClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

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


#create y
y =home_data.SalePrice

# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'BedroomAbvGr','HouseStyle', 
            'TotRmsAbvGrd','GarageCars','KitchenAbvGr','FullBath','GrLivArea','MSZoning',
            'TotalBsmtSF','GarageYrBlt','GarageType','ScreenPorch','SaleCondition','BsmtUnfSF','HouseStyle',
            'RoofStyle','Neighborhood','BsmtFinSF1','BsmtQual','LowQualFinSF','FireplaceQu','Fireplaces',
            'YearRemodAdd','OverallCond','OverallQual','YrSold']


X = home_data[features]

#dealing with missing data 

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer(strategy='constant') #constant is the best strategy to be used here




# Split into validation and training data
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

#to deal with categorical data, get_dummies
one_hot_encoded_training_predictors = pd.get_dummies(train_X)
one_hot_encoded_test_predictors = pd.get_dummies(val_X)
final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)



# print("Unique values in 'Condition2' column in training data:", X_train['Condition2'].unique())
# print("\nUnique values in 'Condition2' column in validation data:", X_valid['Condition2'].unique())



#imputing missing values
imputed_X_train = my_imputer.fit_transform(final_train)
imputed_X_test = my_imputer.transform(final_test)


# Specify Model
iowa_model = DecisionTreeRegressor(random_state=1)


# Fit Model
iowa_model.fit(imputed_X_train, train_y)



# Make validation predictions and calculate mean absolute error
val_predictions = iowa_model.predict(imputed_X_test)
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))



# Using best value for max_leaf_nodes
iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
iowa_model.fit(imputed_X_train, train_y)
val_predictions = iowa_model.predict(imputed_X_test)

#getiing mean absolute error
val_mae = mean_absolute_error(val_predictions, val_y)
print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))



#Logistic regressor (it takes much time and didn't produce good result)
# from sklearn.linear_model import LogisticRegression
# lr_model=LogisticRegression().fit(imputed_X_train,train_y)
# lr_predictions=lr_model.predict(imputed_X_test)
# lr_mae=mean_absolute_error(lr_predictions,val_y)
# print('validation mae for LR model: {:,.0f}'.format(lr_mae))


# Random forest regressor
rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(imputed_X_train, train_y)
rf_val_predictions = rf_model.predict(imputed_X_test)
rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))



#XGBREGRESSOR model
xgbr= XGBRegressor(n_estimators=1150,learning_rate=0.05)
xgbr.fit(imputed_X_train, train_y)
predicts=xgbr.predict(imputed_X_test)
xgbr_mae=mean_absolute_error(predicts,val_y)
print("Validation MAE for XGBRegressor Model: {:,.0f}".format(xgbr_mae))


#dealing with missing values and categorical data
one_hot_encoded_training_predictors = pd.get_dummies(X)
imputed_X_train = my_imputer.fit_transform(one_hot_encoded_training_predictors)

#defining a XGBR model

xgbr= XGBClassifier(n_estimators=1000,learning_rate=0.05)
xgbr.fit(imputed_X_train, y)

# path to file you will use for predictions
test_data_path = '../input/test.csv'

# read test data file using pandas
test_data = pd.read_csv(test_data_path)

# create test_X which comes from test_data but includes only the columns you used for prediction.
# The list of columns is stored in a variable called features
test_X = test_data[features]


#dealing with categorical values
one_hot_encoded_training_predictors = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(test_X)

final_train, final_test = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors,
                                                                    join='left', 
                                                                    axis=1)

#dealing with missing values
imputed_X_train = my_imputer.fit_transform(final_train)
imputed_X_test = my_imputer.transform(final_test)
# fit rf_model_on_full_data on all data from the training data
xgbr.fit(imputed_X_train,y)



# make predictions which we will submit. 
predicts=xgbr.predict(imputed_X_test)

#submit


output = pd.DataFrame({'Id': test_data.Id,
                      'SalePrice': predicts})
output.to_csv('submission.csv', index=False)