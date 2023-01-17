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
train_ds = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test_ds  = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
train_ds.head()
test_ds.head()
train_ds.columns
object_list = list(train_ds.select_dtypes(include=['object']).columns)

object_list
dummies = pd.get_dummies(train_ds[object_list])
train_ds = train_ds.drop(object_list, axis=1)

train_ds = pd.concat([train_ds, dummies], axis=1)
train_ds.isna().sum().sort_values()
train_ds['LotFrontage'].fillna((train_ds['LotFrontage'].mean()), inplace=True)

train_ds['GarageYrBlt'].fillna((train_ds['GarageYrBlt'].mean()), inplace=True)

train_ds['MasVnrArea'].fillna((train_ds['MasVnrArea'].mean()), inplace=True)



train_ds.sample()
train_ds.isna().sum().sort_values()
train_ds = train_ds.drop(['Utilities_NoSeWa', 'Condition2_RRAe', 'Condition2_RRAn',

       'Condition2_RRNn', 'HouseStyle_2.5Fin', 'RoofMatl_ClyTile',

       'RoofMatl_Membran', 'RoofMatl_Metal', 'RoofMatl_Roll',

       'Exterior1st_ImStucc', 'Exterior1st_Stone', 'Exterior2nd_Other',

       'Heating_Floor', 'Heating_OthW', 'Electrical_Mix', 'GarageQual_Ex',

       'PoolQC_Fa', 'MiscFeature_TenC'], axis=1)

train_ds.sample(5)
from sklearn.model_selection import train_test_split
x = train_ds.drop(['SalePrice'], axis=1).values

y = train_ds.SalePrice.values
x[0]
y
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 10)
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error, r2_score

regressor = LinearRegression()

regressor.fit(x_train, y_train)
y_pred = regressor.predict(x_test)
print('mean_squared_error: ', mean_squared_error(y_test, y_pred))

print('r2_score:\t',r2_score(y_test, y_pred))
test_ds.isna().sum().sort_values()
object_list = list(test_ds.select_dtypes(include=['object']).columns)

dummies = pd.get_dummies(test_ds[object_list])



test_ds = test_ds.drop(object_list, axis=1)

test_ds = pd.concat([test_ds, dummies], axis=1)



test_ds.isna().sum().sort_values()
test_ds['LotFrontage'].fillna((test_ds['LotFrontage'].mean()), inplace=True)

test_ds['GarageYrBlt'].fillna((test_ds['GarageYrBlt'].mean()), inplace=True)

test_ds['MasVnrArea'].fillna((test_ds['MasVnrArea'].mean()), inplace=True)

test_ds['BsmtHalfBath'].fillna((test_ds['BsmtHalfBath'].mean()), inplace=True)

test_ds['BsmtFullBath'].fillna((test_ds['BsmtFullBath'].mean()), inplace=True)

test_ds['TotalBsmtSF'].fillna((test_ds['TotalBsmtSF'].mean()), inplace=True)

test_ds['GarageArea'].fillna((test_ds['GarageArea'].mean()), inplace=True)

test_ds['GarageCars'].fillna((test_ds['GarageCars'].mean()), inplace=True)

test_ds['BsmtFinSF2'].fillna((test_ds['BsmtFinSF2'].mean()), inplace=True)

test_ds['BsmtFinSF1'].fillna((test_ds['BsmtFinSF1'].mean()), inplace=True)

test_ds['BsmtUnfSF'].fillna((test_ds['BsmtUnfSF'].mean()), inplace=True)



test_ds.isna().sum().sort_values()
x_test.shape
test = test_ds.values

test.shape
y_pred = regressor.predict(test)
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = y_pred

sub.to_csv('output.csv', index=False)