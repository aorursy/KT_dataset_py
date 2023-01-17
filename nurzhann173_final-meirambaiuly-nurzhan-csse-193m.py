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
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.head()
test.head()
test.info()
test.describe()
missing = train.isnull().sum()

missing = missing[missing > 0]

missing.sort_values(inplace=True)

missing.plot.bar()
train.shape
train['PoolQC'].isnull().sum(),train['MiscFeature'].isnull().sum(),train['Alley'].isnull().sum()
train.drop(['Alley'],axis=1,inplace=True)

train.drop(['MiscFeature'],axis =1,inplace=True)

train.drop(['PoolQC'],axis=1,inplace=True)

train.drop(['Fence'],axis=1,inplace=True)

train.drop(['GarageYrBlt'],axis=1,inplace=True)
train.drop(['FireplaceQu'],axis=1,inplace=True)
train.shape
train['LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean())

train['BsmtCond']=train['BsmtCond'].fillna(train['BsmtCond'].mode()[0])

train['BsmtQual']=train['BsmtQual'].fillna(train['BsmtQual'].mode()[0])
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import cross_val_score

from sklearn.impute import SimpleImputer

#Divide into predictor and target variables

train_X = train.drop('SalePrice', axis=1)

train_y = train.SalePrice

test_X = test
dum_train_X = pd.get_dummies(train_X)

dum_test_X = pd.get_dummies(test_X)

train_X, test_X = dum_train_X.align(dum_test_X, join='left', axis=1)

my_imputer = SimpleImputer()

train_X = my_imputer.fit_transform(train_X)

test_X = my_imputer.transform(test_X)

reg = LinearRegression()

cv_scores = cross_val_score(reg, train_X, train_y, cv=5)

print(cv_scores)
reg = LinearRegression()

reg.fit(train_X, train_y)

predictions = reg.predict(test_X)
predictions
submission_linreg = pd.DataFrame({'Id': test.Id, 'SalePrice':predictions})
submission_linreg.to_csv('submission_linear_reg.csv', index=False)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 10, random_state = 0)

regressor.fit(train_X, train_y)

randomForestPredictions = regressor.predict(test_X)
randomForestPredictions
submission_randomForest = pd.DataFrame({'Id': test.Id, 'SalePrice':randomForestPredictions})
submission_linreg.to_csv('submission_random_forest.csv', index=False)