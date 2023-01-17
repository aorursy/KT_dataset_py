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
with open('/kaggle/input/house-prices-advanced-regression-techniques/data_description.txt', 'r') as f:

    for line in f.readlines():

        if line[0] != ' ':

            print(line)
import matplotlib.pyplot as plt

import seaborn as sns

import time

import random

import missingno as msno

import warnings

warnings.filterwarnings('ignore')
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')



train.head(3)
len(train.columns)
train['SalePrice'][:5]
size_issue = ['LotFrontage', 'LotArea', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'GarageArea', 'WoodDeckSF', 

             'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']



type_issue = ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2',

             'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 

             'Heating', 'CentralAir', 'Electrical', 'Functional', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature', 'SaleType', ]



count_issue = ['BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',  ]



time_issue = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'MoSold', 'YrSold']



quality_issue = ['OverallQual', 'OverallCond', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond', 'HeatingQC', 'KitchenQual', 'FireplaceQu', 'GarageQual', 'GarageCond', 

                'PoolQC', 'Fence', 'SaleCondition']



value_issue = ['MiscVal']



len(size_issue) + len(type_issue) + len(count_issue) + len(time_issue) + len(quality_issue) + len(value_issue)
train['OverallQual'].describe()
train['OverallCond'].describe()
train['Fence'].describe()
set(train[train['Fence'].isna()==False]['Fence'])
train['LotArea'].describe()
train[size_issue].head(5)
train[type_issue].head()
train[time_issue].head()
train[quality_issue].head(5)
train[count_issue].head()
train[value_issue].head()
msno.matrix(train[size_issue])
plt.figure(figsize=(16,9))

plt.scatter(train['LotArea'], train['SalePrice'], color='royalblue', s=10)

plt.xlim(0, 50000)
plt.figure(figsize=(16,9))

plt.scatter(train['GrLivArea'], train['SalePrice'], color='royalblue', s=10)

plt.xlim(0, 5000)
plt.figure(figsize=(16,9))

plt.scatter(train['PoolArea'], train['SalePrice'], color='royalblue', s=10)

plt.xlim(0, 1000)
plt.figure(figsize=(16,9))

plt.scatter(train['GarageArea'], train['SalePrice'], color='royalblue', s=10)

plt.xlim(0, 1500)
plt.figure(figsize=(16,9))

plt.scatter(train['TotalBsmtSF'], train['SalePrice'], color='royalblue', s=10)

plt.xlim(0, 3000)
train['totalArea'] = train['TotalBsmtSF'] + train['GrLivArea']

plt.figure(figsize=(16,9))

plt.scatter(train['totalArea'], train['SalePrice'], color='royalblue', s=10)

plt.xlim(0, 7000)
train['totalArea'] = train['TotalBsmtSF'] + train['GrLivArea']

plt.figure(figsize=(16,9))

plt.scatter(train['totalArea'], np.log(train['SalePrice']), color='royalblue', s=10)

plt.xlim(0, 7000)
train['MSSubClass'].describe()
train.groupby('MSSubClass')['SalePrice'].mean().plot(kind='bar', figsize=(12,6.6), color='dodgerblue', edgecolor='k')
train.groupby('Neighborhood')['SalePrice'].mean().plot(kind='bar', figsize=(12,6.6), color='dodgerblue', edgecolor='k')
train.groupby('LandSlope')['SalePrice'].mean().plot(kind='bar', figsize=(12,6.6), color='dodgerblue', edgecolor='k')
train.groupby('SaleType')['SalePrice'].mean().plot(kind='bar', figsize=(12,6.6), color='dodgerblue', edgecolor='k')
train.groupby('Heating')['SalePrice'].mean().plot(kind='bar', figsize=(12,6.6), color='dodgerblue', edgecolor='k')
train[time_issue].head(2)
train.groupby('MoSold')['SalePrice'].mean().plot(kind='bar', color='tomato', edgecolor='darkred', figsize=(12, 6.6))
train.groupby('YrSold')['SalePrice'].mean().plot(kind='bar', color='tomato', edgecolor='darkred', figsize=(12, 6.6))
train['SinceBuilt'] = train['YrSold'] - train['YearBuilt']

train['SinceRemod'] = train['YrSold'] - train['YearRemodAdd']



train.groupby('SinceBuilt')['SalePrice'].mean().plot(kind='bar', color='tomato', edgecolor='darkred', figsize=(16, 6.6))
train.groupby('SinceRemod')['SalePrice'].mean().plot(kind='bar', color='tomato', edgecolor='darkred', figsize=(16, 6.6))
train[quality_issue].head(2)
train.groupby('OverallQual')['SalePrice'].mean().plot(kind='bar', color='tan', edgecolor='k', figsize=(12, 6.6))
train.groupby('OverallCond')['SalePrice'].mean().plot(kind='bar', color='tan', edgecolor='k', figsize=(12, 6.6))
train.groupby('ExterQual')['SalePrice'].mean().plot(kind='bar', color='tan', edgecolor='k', figsize=(12, 6.6))
train.groupby('BsmtQual')['SalePrice'].mean().plot(kind='bar', color='tan', edgecolor='k', figsize=(12, 6.6))
train.groupby('HeatingQC')['SalePrice'].mean().plot(kind='bar', color='tan', edgecolor='k', figsize=(12, 6.6))
train.groupby('KitchenQual')['SalePrice'].mean().plot(kind='bar', color='tan', edgecolor='k', figsize=(12, 6.6))
train[count_issue].head(2)
train.groupby('TotRmsAbvGrd')['SalePrice'].mean().plot(kind='bar', color='paleturquoise', edgecolor='teal', figsize=(12, 6.6))
train.groupby('BedroomAbvGr')['SalePrice'].mean().plot(kind='bar', color='paleturquoise', edgecolor='teal', figsize=(12, 6.6))
train.groupby('GarageCars')['SalePrice'].mean().plot(kind='bar', color='paleturquoise', edgecolor='teal', figsize=(12, 6.6))
train.groupby('FullBath')['SalePrice'].mean().plot(kind='bar', color='paleturquoise', edgecolor='teal', figsize=(12, 6.6))
train[value_issue].tail(2)
train.MiscVal.describe()
plt.figure(figsize=(12,6.6))

plt.scatter(train['MiscVal'], train.SalePrice, color='thistle', s=7)
my_feat = train[['totalArea', 'MSSubClass', 'Neighborhood', 'SaleType', 'Heating', 'SinceRemod', 'OverallQual', 'ExterQual', 'TotRmsAbvGrd', 'GarageCars', 'SalePrice']]

my_feat.head(5)
from sklearn.preprocessing import LabelEncoder

encod = LabelEncoder()
my_feat['MSSubClass'] = my_feat['MSSubClass'].apply(str)

one = pd.get_dummies(my_feat[['MSSubClass', 'Neighborhood', 'SaleType', 'Heating']])

my_feat.replace('FA', 0).replace('TA', 1).replace('Gd', 2).replace('Ex', 3)

train_df = my_feat.drop(columns=['MSSubClass', 'Neighborhood', 'SaleType', 'Heating', 'ExterQual']).merge(one, how='left', left_index=True, right_index=True)

train_df.head(10)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

temp = scaler.fit_transform(train[['totalArea', 'SinceRemod', 'OverallQual', 'TotRmsAbvGrd', 'GarageCars']])

train_dataset = train_df.drop(columns=['totalArea', 'SinceRemod', 'OverallQual', 'TotRmsAbvGrd', 'GarageCars']).merge(pd.DataFrame(temp), how='left', left_index=True, right_index=True)

train_dataset.head(5)
train_dataset.rename(columns={0: 'totalArea', 1: 'SinceRemod', 2: 'OverallQual', 3:'TotRmsAbvGrd', 4:'GarageCars'}, inplace=True)

train_dataset.head(5)
from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split

from sklearn.model_selection import cross_val_score

from random import randint

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score
k = randint(0, 100)

params = {'n_estimators': [10*i for i in range(1, 15)], 'max_depth': [2*i for i in range(1, 10)]}

# lr = LinearRegression()

rf = RandomForestRegressor()

x_train, x_test, y_train, y_test = train_test_split(train_dataset.drop(columns=['SalePrice']), train_dataset['SalePrice'], random_state = k, test_size=0.25)

# rf.fit(x_train, y_train)

# pred = lr.predict(x_test)

# pred

# rf.score(x_test, y_test)

best_rf = GridSearchCV(rf, param_grid=params, cv=5)

best_rf.fit(x_train, y_train)

pred = best_rf.predict(x_test)

best_rf.score(x_test, y_test)
rf = RandomForestRegressor(n_estimators=100)

k = randint(0, 100)

x_train, x_test, y_train, y_test = train_test_split(train_dataset.drop(columns=['SalePrice']), train_dataset['SalePrice'], random_state = k, test_size=0.25)

rf.fit(x_train, y_train)

rf.score(x_test, y_test)
X = train_dataset.drop(columns=['SalePrice'])

y = train_dataset['SalePrice']

cross_val_score(rf, X, y, cv=5)
test = test.replace(np.nan, 0)

test_feat = test[['MSSubClass', 'Neighborhood', 'SaleType', 'Heating', 'ExterQual', 'OverallQual', 'TotRmsAbvGrd', 'GarageCars']]

test_feat['totalArea'] = test['TotalBsmtSF'] + test['GrLivArea']

test_feat['SinceRemod'] = test['YrSold'] - test['YearRemodAdd']

test_feat.head(5)
test_feat['MSSubClass'] = test_feat['MSSubClass'].apply(str)

one = pd.get_dummies(test_feat[['MSSubClass', 'Neighborhood', 'SaleType', 'Heating']])

test_feat.replace('FA', 0).replace('TA', 1).replace('Gd', 2).replace('Ex', 3)

test_df = test_feat.drop(columns=['MSSubClass', 'Neighborhood', 'SaleType', 'Heating', 'ExterQual']).merge(one, how='left', left_index=True, right_index=True)



scaler = StandardScaler()

temp = scaler.fit_transform(test_feat[['totalArea', 'SinceRemod', 'OverallQual', 'TotRmsAbvGrd', 'GarageCars']])

test_dataset = test_df.drop(columns=['totalArea', 'SinceRemod', 'OverallQual', 'TotRmsAbvGrd', 'GarageCars']).merge(pd.DataFrame(temp), how='left', left_index=True, right_index=True)



test_dataset.head()
test_dataset.rename(columns={0: 'totalArea', 1: 'SinceRemod', 2: 'OverallQual', 3:'TotRmsAbvGrd', 4:'GarageCars'}, inplace=True)

test_dataset.head()
msno.matrix(test_dataset)
test_dataset.describe()
test_dataset.columns
np.any(np.isnan(test_dataset))
np.all(np.isfinite(test_dataset))
test_dataset.isna().sum()
prediction = rf.predict(test_dataset)

prediction[:5]
pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
test['SalePrice'] = prediction

submission = test[['Id', 'SalePrice']]

submission.sample(5)
submission.to_csv('Submission2.csv', index=False)