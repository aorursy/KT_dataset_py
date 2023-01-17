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

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
# read in the data

train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
train.head()
# create training target dataset

train_y = train.SalePrice



# create training features dataset

predictor_cols = ['YearBuilt', 'OverallQual']

train_X = train[predictor_cols]
# fit the features to Linear Regression

lin_reg.fit(train_X, train_y)



# predicting on training data-set

y_train_pred = lin_reg.predict(train_X)



# evaluating the model on training dataset (RMSE)

from math import sqrt

sqrt(mean_squared_error(train_y, y_train_pred))
# Read the test data

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

# Treat the test data in the same way as training data. In this case, pull same columns.

test_X = test[predictor_cols]

# Use the model to make predictions

predicted_prices = lin_reg.predict(test_X)

# We will look at the predicted prices to ensure we have something sensible.

print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})

# you could use any filename. We choose submission here

my_submission.to_csv('submission.csv', index=False)