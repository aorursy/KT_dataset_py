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
# import pandas_profiling as pdp
df = pd.read_csv("../input/train.csv")
pd.options.display.max_rows=5
df
# pdp.ProfileReport(df)
from sklearn.ensemble import RandomForestRegressor
# pull data into target (y) and predictors (X)
train_y = df.SalePrice
# predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
# Create training predictors data
# train_X = df[predictor_cols]
train_X = df.drop(['SalePrice'], 1)
import category_encoders as ce
encoder = ce.OneHotEncoder()
train_X.replace(np.nan, 0, inplace=True)
train_X = encoder.fit_transform(train_X)

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)


test = pd.read_csv('../input/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test.replace(np.nan, 0)
test_X = encoder.transform(test_X)
# test_X = test.replace(np.nan, 0, inplace=True)
# test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission_1723.csv', index=False)
import matplotlib.pyplot as plt
df.SalePrice.hist()
np.log10(df.SalePrice).hist()
