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
train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt

from sklearn.model_selection import train_test_split

import seaborn as sns

from sklearn.linear_model import LogisticRegression



# dropna drops missing values (think of na as "not available")

#data = train.dropna(axis=0)

#test = test.dropna(axis=0)



y = train.SalePrice

# Create X

features = ['LotArea','GrLivArea','OverallQual','GarageArea','TotalBsmtSF','YearBuilt', '1stFlrSF', 'FullBath', 'BedroomAbvGr','TotRmsAbvGrd']

X = train[features]



X.shape





train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)



sns.jointplot(x=train['GrLivArea'], y=train.SalePrice, kind='reg')



sns.jointplot(x=train['LotArea'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['OverallQual'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['GarageArea'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['TotalBsmtSF'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['YearBuilt'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['1stFlrSF'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['FullBath'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['BedroomAbvGr'], y=train.SalePrice, kind='reg')

sns.jointplot(x=train['TotRmsAbvGrd'], y=train.SalePrice, kind='reg')

train_X.isnull().sum()
forest_model = RandomForestRegressor(max_depth=10, max_features='sqrt',

                                   min_samples_leaf=2, min_samples_split=2,random_state=10)

forest_model.fit(train_X, train_y)

preds = forest_model.predict(val_X)

print(sqrt(mean_squared_error(val_y,preds)))



#test_X = pd.concat([test[features],test.Id])

features = ['Id','LotArea','GrLivArea','OverallQual','GarageArea','TotalBsmtSF','YearBuilt', '1stFlrSF', 'FullBath', 'BedroomAbvGr','TotRmsAbvGrd']

test_X = test[features]

#test_X.isnull().sum()

#test_X.shape

test_X = test_X.fillna(test_X.median(), inplace=False)

IDs = test_X['Id']



test_X.drop('Id', axis = 1, inplace = True)

IDs.shape

test_preds = forest_model.predict(test_X)

output = pd.DataFrame({'Id': IDs,'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)

test_X.head()

test_X.shape