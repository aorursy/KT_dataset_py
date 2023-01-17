# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/train.csv')

test = pd.read_csv('../input/test.csv')

ID=test.Id
train.head()
train.info()
train.SalePrice.describe()
sns.boxplot(x = train.MSSubClass, y=train.SalePrice, fliersize=5)
plt.figure(figsize= (10,10))

cormat = train.corr()

sns.heatmap(cormat)
train.isnull().sum();
for x in ("PoolQC", "MiscFeature" , "Alley", "Fence", "FireplaceQu",

         'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',

         'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1',

         "MasVnrType", 'MSSubClass'):

    train[x] = train[x].fillna('None')

    test[x] = test[x].fillna('None')

for x in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath',

         'GarageYrBlt', 'GarageArea', 'GarageCars'):

    train[x] = train[x].fillna(0)

    test[x] = test[x].fillna(0)

for x in train.columns:

    train[x].fillna(train[x].median, inplace= True)

for x in test.columns:

    test[x].fillna(test[x].median, inplace= True)

train.isnull().sum().sum()
test.isnull().sum().sum()
train.dtypes;
train.drop('BsmtFinType2', inplace=True, axis = 1)

test.drop('BsmtFinType2', inplace=True, axis = 1)

train.drop('LotFrontage', inplace=True, axis = 1)

test.drop('LotFrontage', inplace=True, axis = 1)

train.drop('MasVnrArea', inplace=True, axis = 1)

test.drop('MasVnrArea', inplace=True, axis = 1)

train.drop('Electrical', inplace=True, axis = 1)

test.drop('Electrical', inplace=True, axis = 1)

train.drop('MSZoning', inplace=True, axis = 1)

test.drop('MSZoning', inplace=True, axis = 1)
from sklearn.preprocessing import LabelEncoder

#cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 

        #'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 

        # 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',

       # 'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 

       # 'YrSold', 'MoSold', 'SaleType','SaleCondition', 'MiscFeature', 'GarageType', 'Electrical'

       #'Heating', 'Foundation', MasVnrType, Exterior2nd, Exterior1st, RoofMatl,

      # RoofStyle, )

a=0

for x in train.columns.drop('SalePrice'):

    lr = LabelEncoder() 

    lr.fit(train[x]) 

    train[x] = lr.transform(train[x])

for x in test.columns:

    lr = LabelEncoder()

    test[x]=str(test[x])

    lr.fit(test[x]) 

    test[x] = lr.transform(test[x])



train.head()
from sklearn.preprocessing import scale

train_s = pd.DataFrame(scale(train.drop(['SalePrice', 'Id'], axis= 1)))

train_s.columns = train.columns.drop(['SalePrice', 'Id'])

train_s['SalePrice'] = train.SalePrice

train_s.head()
test_s = pd.DataFrame(scale(test.drop("Id", axis=1)))

test_s.columns = test.columns.drop('Id')
train_s.shape
test_s.shape
from sklearn.neighbors import KNeighborsRegressor

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

from sklearn.kernel_ridge import KernelRidge

from sklearn.ensemble import GradientBoostingRegressor
sklearn.metrics.SCORERS.keys()
clf = KNeighborsRegressor(n_neighbors=20,p = 1,weights='distance',metric='minkowski')

clf.fit(train_s.drop("SalePrice", axis= 1), train_s.SalePrice)

clf1 = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)

clf1.fit(train_s.drop("SalePrice", axis= 1), train_s.SalePrice)
from xgboost import XGBRegressor
XGBR = XGBRegressor(n_estimators=200, learning_rate=0.1)

XGBR.fit(train_s.drop("SalePrice", axis= 1), train_s.SalePrice)

kv = KFold(n_splits=5,random_state=42,shuffle=True)

np.mean(cross_val_score(XGBR,train_s.drop("SalePrice", axis= 1),train_s.SalePrice, cv=kv,

                        scoring = 'neg_mean_squared_error' ))
kv = KFold(n_splits=5,random_state=42,shuffle=True)

np.mean(cross_val_score(clf,train_s.drop("SalePrice", axis= 1),train_s.SalePrice, cv=kv,

                        scoring = 'neg_mean_squared_error' ))
kv = KFold(n_splits=5,random_state=42,shuffle=True)

np.mean(cross_val_score(clf1,train_s.drop("SalePrice", axis= 1),train_s.SalePrice, cv=kv,

                        scoring = 'neg_mean_squared_error' ))
kv = KFold(n_splits=5,random_state=42,shuffle=True)

np.mean(cross_val_score(clf2,train_s.drop("SalePrice", axis= 1),train_s.SalePrice, cv=kv,

                        scoring = 'neg_mean_squared_error' ))
preds = XGBR.predict(test_s)
sub = pd.DataFrame()

sub['Id'] = ID

sub['SalePrice'] = preds

sub.to_csv('submission.csv',index=False)