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
# Read data
train_data = pd.read_csv("../input/train.csv")
# Target column
train_y = train_data.SalePrice
# Train features
predict_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']
train_X = train_data[predict_cols]
# Define model
forest_model = RandomForestRegressor()
forest_model.fit(train_X, train_y)
# Test
test_data = pd.read_csv("../input/test.csv")

test_X = test_data[predict_cols]
predicted_prices = forest_model.predict(test_X)
print(predicted_prices)
# Prepare submission file
my_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)