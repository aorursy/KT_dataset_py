# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from sklearn.metrics import mean_squared_error
# read in the data

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
# create training target dataset

y_train = train.SalePrice



# create training features dataset

cols_to_drop = []

X_train = train



# Iterate over each column of X_train

for col in X_train.columns:

    # Check if the column is of int64ype

    if X_train[col].dtype != 'int64':

        # If not. add column to list of columns to drop

        cols_to_drop.append(col)

        

# drop non-numeric type columns

X_train = X_train.drop(cols_to_drop, axis=1)



# drop 'SalePrice' column

X_train = X_train.drop('SalePrice', axis=1)
X_train.columns
from sklearn.ensemble import RandomForestRegressor

rnd_rgr = RandomForestRegressor(n_estimators=500, max_leaf_nodes=500, n_jobs=-1, random_state=42)



# fit the features to Random Forest Classifier

rnd_rgr.fit(X_train, y_train)



# predicting on train dataset

y_pred_rf = rnd_rgr.predict(X_train)
# evaluating the model on training dataset (RMSE)

from math import sqrt

sqrt(mean_squared_error(y_train, y_pred_rf))
# Read the test data

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test.drop(cols_to_drop, axis=1)



test_X.isnull().sum()
# Impute the missing values with mean imputation

test_X.fillna(test_X.median(), inplace=True)



# Count the number of NaNs in the dataset to verify

test_X.isnull().sum()
# Use the model to make predictions

predicted_prices = rnd_rgr.predict(test_X)



# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)