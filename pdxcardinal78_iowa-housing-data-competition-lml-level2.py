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
# Load Data
iowa_data = pd.read_csv('../input/train.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

iowa_target = iowa_data.SalePrice
iowa_predictors = iowa_data.drop(['SalePrice'], axis=1)

# Exclude all but numeric values
iowa_numeric_predictors = iowa_predictors.select_dtypes(exclude=['object'])
# Split data train, test, split
X_train, X_test, y_train, y_test = train_test_split(iowa_numeric_predictors, iowa_target, random_state=0)
# score_dataset function
def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictors = model.predict(X_test)
    return mean_absolute_error(y_test, predictors)
    
# Determine MAE score when columns are dropped
cols_with_missing = [col for col in X_train.columns
                         if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values: ")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))
from sklearn.preprocessing import Imputer

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print('Mean Absolute Error from Imputing:')
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))
# Find score with extra columns showing imputation

imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing_plus = (col for col in X_train.columns
                                    if X_train[col].isnull().any())
for col in cols_with_missing_plus:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()
    
# Impute it
my_imputer = Imputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print('Mean Absolute Error from IMputation while Tracking what was imputed:')
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))
# Create New Prediction with Imputer values
test = pd.read_csv('../input/test.csv')
test_X = test.select_dtypes(exclude=['object'])
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(test_X)
# Create Model
my_model = RandomForestRegressor()
my_model.fit(imputed_X_train, y_train)
predictors = my_model.predict(imputed_X_test)
# send results to CSV Files
test_submission_2 = pd.DataFrame({'Id': test.Id, 'SalePrice': predictors}).to_csv('submission_2.csv', index=False)
