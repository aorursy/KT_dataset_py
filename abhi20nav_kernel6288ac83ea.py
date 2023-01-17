# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import os
iowa_filepath = '../input/train.csv'
test_iowa_filepath = '../input/test.csv'
iowa_data = pd.read_csv(iowa_filepath)
test_iowa_data = pd.read_csv(test_iowa_filepath)



# Any results you write to the current directory are saved as output.
y = iowa_data.SalePrice
cols_for_model = ['LotArea','LotFrontage','OverallQual','OverallCond','GrLivArea','FullBath','HalfBath','GarageArea','WoodDeckSF','OpenPorchSF']


import sys
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
X = iowa_data[cols_for_model]
test_X = test_iowa_data[cols_for_model]
train_X, val_X, train_y, val_y = train_test_split(X.as_matrix(),y.as_matrix(),test_size=0.20)
final_test_X = test_X.as_matrix()






from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
train_X = my_imputer.fit_transform(train_X)
final_test_X = my_imputer.transform(final_test_X)


from xgboost import XGBRegressor
my_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)
my_model.fit(train_X, train_y, early_stopping_rounds=5, 
             eval_set=[(val_X, val_y)], verbose=False)
predictions = my_model.predict(final_test_X)
print(predictions)

my_submission = pd.DataFrame({'Id': test_iowa_data.Id, 'SalePrice':predictions})

my_submission.to_csv('submission.csv', index=False)