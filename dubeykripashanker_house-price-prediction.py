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
# reading input datasets train test and sample submission
train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')
sample_result = pd.read_csv('../input/sample_submission.csv')

# data analysis for train data
print(train_data.head())
print(train_data.describe())
print(train_data.columns)
print(train_data.dtypes)
# dropna drops missing values (think of na as "not available")
# Just have variables should be in the final model before we drop NAs
var_interest = ['SalePrice','LotArea', 'OverallQual', 'OverallCond','YearBuilt','MiscVal']
train_data_new = train_data[var_interest]
train_data_new = train_data_new.dropna(axis=0)
#Taking response y variable from train data set
y_data = train_data_new['SalePrice']
# Sumarising Train data it looks right skewed having very large difference between 3rd quantile and maximum and mean>medium  
y_data.describe()
# regressor or predictor variable 
x_vars = ['LotArea', 'OverallQual', 'OverallCond','MiscVal']
x_data = train_data_new[x_vars]
x_data.describe()

#test data
test_data_new = test_data[x_vars]
test_data_new = test_data_new.dropna(axis = 0)

#Running model
from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
house_price_model = DecisionTreeRegressor(random_state=1)

# Fit model
house_price_model.fit(x_data, y_data)

#In Sample check
print("Making predictions for the following 5 houses:")
print(x_data.head())
print(y_data.head())
print("The predictions are")
print(house_price_model.predict(x_data.head()))

#Test model using test data
print("The test data predictions are")
sale_price_test = house_price_model.predict(test_data_new)
print(sale_price_test)
# Prepare Submission File
ks_submission = pd.DataFrame({'Id': test_data.Id, 'SalePrice': sale_price_test})
# you could use any filename. We choose submission here
ks_submission.to_csv('submission.csv', index=False)


