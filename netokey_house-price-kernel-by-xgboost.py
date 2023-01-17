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
# Read Data
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
# Data Exploration
train_data.SalePrice.isnull().any()
# get the object columns and the number unique value is above 10
cols_object_too_many_unique = [col for col in train_data.columns 
                               if train_data[col].dtype=='object'
                               and train_data[col].nunique()>10]
# get the object columns with missing value in the rest columns
# cols_object_with_missing = [col for col in train_data.columns
#                             if train_data[col].dtype=='object'
#                            and train_data[col].nunique()<=10
#                            and train_data[col].isnull().any()]
# get the columns to drop
cols_to_drop = cols_object_too_many_unique # + cols_object_with_missing
# Train data, target data and test data
# Target column
train_y = train_data.SalePrice
# Train features
# drop the columns
prepared_train_data = train_data.drop(['Id','SalePrice']+cols_to_drop, axis=1)
prepared_test_data = test_data.drop(['Id']+cols_to_drop, axis=1)

# extension columns for NaN
cols_missing_train = [col for col in prepared_train_data.columns
                      if prepared_train_data[col].isnull().any()
                      and prepared_train_data[col].dtype != 'object'
                     ]
cols_missing_test = [col for col in prepared_test_data.columns
                      if prepared_test_data[col].isnull().any()
                      and prepared_test_data[col].dtype != 'object'
                     ]
for col in cols_missing_test:
    prepared_train_data[col+"_the_missing"]=prepared_train_data[col].isnull()
    prepared_test_data[col+"_the_missing"]=prepared_test_data[col].isnull()

# one hot encode
one_hot_encode_train_X = pd.get_dummies(prepared_train_data, dummy_na=True)
one_hot_encode_test_X = pd.get_dummies(prepared_test_data, dummy_na=True)
dummied_train_X, dummied_test_X = one_hot_encode_train_X.align(one_hot_encode_test_X, join='left', axis=1)

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
train_X = my_imputer.fit_transform(dummied_train_X)
test_X = my_imputer.transform(dummied_test_X)
# split for XGBoost
from sklearn.model_selection import train_test_split
XGB_train_X, XGB_test_X, XGB_train_y, XGB_test_y = train_test_split(dummied_train_X.as_matrix(),
                                                                    train_y.as_matrix(),
                                                                    test_size=0.25)
XGB_imputer = Imputer()
XGB_train_X = XGB_imputer.fit_transform(XGB_train_X)
XGB_test_X = XGB_imputer.transform(XGB_test_X)
# XGBoost Model test
from xgboost import XGBRegressor
XGB_model = XGBRegressor(n_estimators=300,learning_rate=0.05)
XGB_model.fit(XGB_train_X, XGB_train_y, early_stopping_rounds=5, 
              eval_set=[(XGB_test_X, XGB_test_y)], verbose=False)
predictions = XGB_model.predict(XGB_test_X)
# get MAE
from sklearn.metrics import mean_absolute_error

print("Mean Absolute Error : " + str(mean_absolute_error(predictions, XGB_test_y)))
# Define model
model = XGBRegressor(n_estimators=300,learning_rate=0.05)
model.fit(train_X, train_y, verbose=False)
# predict test
predicted_prices = model.predict(test_X)
# Prepare submission file
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)