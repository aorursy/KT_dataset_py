# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
#from sklearn.preprocessing import Imputer      Decepreted and will be removed in version 0.22
#hence used SimpleImputer

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#importing training and test data
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
y = train.SalePrice
X = train.drop('SalePrice', axis=1)
X = X.select_dtypes(exclude=['object'])

train_X, val_X, train_y, val_y = train_test_split(X,
                                                 y,
                                                 train_size=0.7,
                                                 test_size=0.3,
                                                 random_state=0)
def mae_score(train_X, val_X, train_y, val_y):
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    predict = model.predict(val_X)
    mae = mean_absolute_error(val_y, predict)
    return mae
#Method 1 for handling NaN values : Drop
NaN_columns = [col for col in train_X.columns
                           if train_X[col].isnull().any()]
reduced_train_X = train_X.drop(NaN_columns, axis=1)
reduced_val_X = val_X.drop(NaN_columns, axis=1)
mae_drop = mae_score(reduced_train_X, reduced_val_X, train_y, val_y)
print("Mean Absolute Error from Dropping NaN values:")
print(mae_drop)
#Method 2 for handling NaN values : Imputation
my_imputer = SimpleImputer()
imputed_train_X = my_imputer.fit_transform(train_X)
imputed_val_X = my_imputer.fit_transform(val_X)
mae_imputed = mae_score(imputed_train_X, imputed_val_X, train_y, val_y)
print("Mean Absolute Error from Imputation:")
print(mae_imputed)
imputed_train_X_plus = train_X.copy()
imputed_val_X_plus = val_X.copy()

cols_with_missing = (col for col in train_X.columns 
                                 if train_X[col].isnull().any())
for col in cols_with_missing:
    imputed_train_X_plus[col + '_was_missing'] = imputed_train_X_plus[col].isnull()
    imputed_val_X_plus[col + '_was_missing'] = imputed_val_X_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_train_X_plus = my_imputer.fit_transform(imputed_train_X_plus)
imputed_val_X_plus = my_imputer.transform(imputed_val_X_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(mae_score(imputed_train_X_plus, imputed_val_X_plus, train_y, val_y))
#for this notebook we will only consider columns with numeric values
#Excluding columns with dtype: 'object'
test_num = test.select_dtypes(exclude=['object'])

# print(test_num.isnull().sum())  check test data for NaN values
#since test data contains NaN values we will use imputation
my_imputer = SimpleImputer()
X_imputed = my_imputer.fit_transform(X)
test_imputed = my_imputer.fit_transform(test_num)
#prediction on test data for submitting
model = RandomForestRegressor()
model.fit(X_imputed, y)
prediction = model.predict(test_imputed)
print(prediction)
submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': prediction})
submission.to_csv('submission.csv', index=False)
