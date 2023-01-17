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
# os.listdir("../input/house-prices-advanced-regression-techniques")
train=pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test=pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")

train
corr=train.corr()

corr[corr['SalePrice']>0.3].index
train = train[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]

test=test[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF']]
total = train.isnull().sum().sort_values(ascending=False)

percent = (train.isnull().sum()/train.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
total_test = test.isnull().sum().sort_values(ascending=False)

percent_test = (test.isnull().sum()/test.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total_test, percent_test], axis=1, keys=['Total', 'Percent'])

missing_data.head(25)
test=test.drop(missing_data[missing_data['Total']>78].index,1)

test.isna().sum().sort_values(ascending=False)

categorical_feature_mask = train.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols = train.columns[categorical_feature_mask].tolist()
categorical_feature_mask_test = test.dtypes==object

# filter categorical columns using mask and turn it into alist

categorical_cols_test = test.columns[categorical_feature_mask_test].tolist()
# train['GarageYrBlt'] = train['GarageYrBlt'].fillna(train['GarageYrBlt'].mean())

# train['MasVnrArea'] = train['MasVnrArea'].fillna(train['MasVnrArea'].mean())

# train.fillna(train.mean())

# 
x=train[['LotFrontage', 'OverallQual', 'YearBuilt', 'YearRemodAdd', 'MasVnrArea',

       'BsmtFinSF1', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea',

       'FullBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageYrBlt', 'GarageCars',

       'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'SalePrice']]



y=train[['SalePrice']]

y.shape
x.fillna(0,inplace=True)

y.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split

train_x,test_x,train_y,test_y =train_test_split(x,y,test_size=0.1,random_state=101)
from sklearn.metrics import accuracy_score , mean_squared_error

from sklearn.tree import DecisionTreeRegressor

dt=DecisionTreeRegressor()

dt.fit(train_x,train_y)
predict=dt.predict(test_x)

accuracy_score(test_y,predict)