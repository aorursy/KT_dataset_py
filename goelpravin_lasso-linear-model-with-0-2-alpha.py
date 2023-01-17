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
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
# Path of the file to read
trainingDataPath = '../input/train.csv'
testDataPath = '../input/test.csv'

training_data = pd.read_csv(trainingDataPath)
test_data = pd.read_csv(testDataPath)
print(training_data.describe())
# Create target object and call it y
y = training_data.SalePrice
# Create X
features = ['LotArea', 'Neighborhood','OverallQual','OverallCond','SaleCondition','SaleType',
            'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
train_X = training_data[features]
oneHottrain_X = pd.get_dummies(train_X)
val_X = test_data[features]
oneHotval_X = pd.get_dummies(val_X)
train_y = training_data.SalePrice

# Specify Model
# rf_model = RandomForestRegressor(random_state=1)
lasso_model = linear_model.Lasso(alpha=0.2,random_state=1,max_iter=10000)
# Fit Model
# rf_model.fit(oneHottrain_X, train_y)
lasso_model.fit(oneHottrain_X, train_y)
# Make validation predictions and calculate mean absolute error
# val_predictions = rf_model.predict(oneHotval_X)
val_predictions = lasso_model.predict(oneHotval_X)

my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': val_predictions})
my_submission.to_csv('submission.csv', index=False)
print("submission csv ready")