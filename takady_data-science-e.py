# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 2000)



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
house_train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

house_test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

houseData = pd.concat([house_train, house_test], sort=False)

house = house_train.copy()
house.describe
houseData.columns
houseData.head()
print(len(house_train), len(house_test))
plt.figure(figsize=(30,30))

corr = house_train.corr()

ax = sns.heatmap(

    corr, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
house_delete_columns=['Id', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street',

       'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig',

       'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType',

       'HouseStyle', 'OverallQual', 'OverallCond',

       'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',

       'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',

       'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1',

       'BsmtFinType2', 'BsmtFinSF2', 'BsmtUnfSF', 'Heating',

       'HeatingQC', 'CentralAir', 'Electrical', 

       'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath',

       'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual',

       'TotRmsAbvGrd', 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType',

       'GarageYrBlt', 'GarageFinish', 'GarageCars', 'GarageArea', 'GarageQual',

       'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF',

       'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'PoolQC',

       'Fence', 'MiscFeature', 'MiscVal', 'MoSold', 'YrSold', 'SaleType',

       'SaleCondition']
house_train = house_train.drop(house_delete_columns, axis=1)

house_train
house_Built = house_train['YearRemodAdd'] - house_train['YearBuilt']

house_Built[house_Built<0].sum()
house_train.isnull().sum().sort_values(ascending=False)
house_train['SalePrice']
plt.figure(figsize=(10,5))

sns.distplot(house_train['SalePrice'], label='Distribution of Sale Price')

plt.show()
plt.figure(figsize=(10,8))

sns.distplot(house_train['1stFlrSF'])
plt.figure(figsize=(10,8))

sns.distplot(house_train['2ndFlrSF'])
plt.figure(figsize=(10,8))

sns.distplot(house_train['TotalBsmtSF'])
plt.figure(figsize=(10,8))

house_train['TotalSF'] = house_train['1stFlrSF'] + house_train['2ndFlrSF'] + house_train['TotalBsmtSF']

sns.distplot(house_train['TotalSF'])
plt.figure(figsize=(10,8))

sns.distplot(house_train['YearBuilt'])
plt.figure(figsize=(10,8))

sns.distplot(house_train['YearRemodAdd'])
plt.figure(figsize=(20,10))

house_train['Year'] = house_train['YearRemodAdd'] - house_train['YearBuilt']

sns.distplot(house_train['Year'])
plt.figure(figsize=(15,15))

house_train_dropped1 = house_train.drop(['YearBuilt','YearRemodAdd','1stFlrSF','2ndFlrSF','TotalBsmtSF'], axis = 1)

corr1 = house_train_dropped1.corr()

ax = sns.heatmap(

    corr1, 

    vmin=-1, vmax=1, center=0,

    cmap=sns.diverging_palette(20, 220, n=200),

    square=True

)

ax.set_xticklabels(

    ax.get_xticklabels(),

    rotation=45,

    horizontalalignment='right'

);
plt.figure(figsize=(15,10))

sns.scatterplot(x=house_train['Year'],y=house_train['SalePrice'])
plt.figure(figsize=(15,10))

house_train = house_train.drop(house_train[(house_train['Year'] > 100)&(house_train['SalePrice'] > 200000)].index)

sns.scatterplot(x=house_train.Year,y=house_train.SalePrice)
plt.figure(figsize=(15, 10))

sns.scatterplot(x=house_train['TotalSF'],y=house_train['SalePrice'])
plt.figure(figsize=(15, 10))

house_train = house_train.drop(house_train[house_train['TotalSF'] > 7000].index)

sns.scatterplot(x=house_train['TotalSF'],y=house_train['SalePrice'])
house_test = house_test[['YearBuilt','YearRemodAdd','TotalBsmtSF','1stFlrSF','2ndFlrSF']]

house_test
house_test.isnull().sum()
na_columns2 = house_test.isnull().sum()[house_test.isnull().sum()>0].index.tolist()

house_test[na_columns2].dtypes.sort_values()
house_test['TotalBsmtSF'].fillna((house_test['TotalBsmtSF'].mean()), inplace=True)
house_test.isnull().sum()
house_test['TotalSF'] = house_test['1stFlrSF'] + house_test['2ndFlrSF'] + house_test['TotalBsmtSF']

house_test['Year'] = house_test['YearRemodAdd'] - house_test['YearBuilt']
from sklearn import linear_model

from sklearn.model_selection import train_test_split



X_house_train = house_train[['Year', 'TotalSF']]

y_house_train = house_train['SalePrice']

X_house_test = house_test[['Year', 'TotalSF']]
model = linear_model.LinearRegression()

model.fit(X_house_train,y_house_train)



y_house_train_pred = model.predict(X_house_train)

y_house_test_pred = model.predict(X_house_test)
print(y_house_test_pred)
print((X_house_test.size), (y_house_test_pred.size))
plt.scatter(house_test['TotalSF'], y_house_test_pred)
plt.scatter(house_test['Year'], y_house_test_pred)
sub = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

sub['SalePrice'] = list(map(int, y_house_test_pred))

sub.to_csv('submission.csv', index=False)