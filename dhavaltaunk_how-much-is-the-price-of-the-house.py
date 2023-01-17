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
from xgboost import XGBRegressor
import seaborn as sns
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
submit = pd.read_csv('../input/sample_submission.csv')
train.head()
x_train = train[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','CentralAir','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']]
x_test = test[['MSSubClass','LotFrontage','LotArea','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','CentralAir','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','FullBath','HalfBath','BedroomAbvGr','KitchenAbvGr','TotRmsAbvGrd','Fireplaces','GarageYrBlt','GarageCars','GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','MiscVal','MoSold','YrSold']]
x_train['LotFrontage'] = x_train['LotFrontage'].fillna(x_train['LotFrontage'].mean())
x_test['LotFrontage'] = x_test['LotFrontage'].fillna(x_train['LotFrontage'].mean())
x_train['MasVnrArea'] = x_train['MasVnrArea'].fillna(x_train['MasVnrArea'].mean())
x_test['MasVnrArea'] = x_test['MasVnrArea'].fillna(x_train['MasVnrArea'].mean())
x_train['GarageYrBlt'] = x_train['GarageYrBlt'].fillna(x_train['GarageYrBlt'].mean())
x_test['GarageYrBlt'] = x_test['GarageYrBlt'].fillna(x_train['GarageYrBlt'].mean())
x_train['CentralAir'] = x_train['CentralAir'].replace({'Y','N'},{1,0})
x_test['CentralAir'] = x_test['CentralAir'].replace({'Y','N'},{1,0})
x_test['TotalBsmtSF'] = x_test['TotalBsmtSF'].fillna(x_test['TotalBsmtSF'].mean())
x_test['BsmtFullBath'] = x_test['BsmtFullBath'].fillna(x_test['BsmtFullBath'].mean())
x_test['BsmtHalfBath'] = x_test['BsmtHalfBath'].fillna(x_test['BsmtHalfBath'].mean())
x_test['GarageCars'] = x_test['GarageCars'].fillna(x_test['GarageCars'].mean())
x_test['GarageArea'] = x_test['GarageArea'].fillna(x_test['GarageArea'].mean())
x_test['BsmtFinSF2'] = x_test['BsmtFinSF2'].fillna(x_test['BsmtFinSF2'].mean())
x_test['BsmtUnfSF'] = x_test['BsmtUnfSF'].fillna(x_test['BsmtUnfSF'].mean())
x_train['Basement'] = x_train['BsmtFinSF2']+x_train['BsmtUnfSF']+x_train['TotalBsmtSF']
x_test['Basement'] = x_test['BsmtFinSF2']+x_test['BsmtUnfSF']+x_test['TotalBsmtSF']
x_train = x_train.drop(['BsmtFinSF2','BsmtUnfSF','TotalBsmtSF'],axis=1)
x_test = x_test.drop(['BsmtFinSF2','BsmtUnfSF','TotalBsmtSF'],axis=1)
y_train = train['SalePrice']
x_train.info()
x_test.info()
X_train = np.array(x_train)
X_test = np.array(x_test)
#from sklearn.decomposition import PCA
#pca = PCA(n_components=24)
#pca.fit(x_train)
#pca.fit_transform(x_train)
#pca.fit_transform(x_test)
#reg = RandomForestRegressor(max_depth=25, n_estimators=128, min_samples_split=12)
reg = XGBRegressor(n_estimators=3500, max_depth=12, learning_rate=0.003)
reg.fit(X_train,y_train)
reg.score(X_train,y_train)
output = pd.DataFrame()
output['Id'] = test['Id']
output['SalePrice'] = reg.predict(X_test)
output.to_csv('output.csv',index=False)
reg.score(X_test, submit['SalePrice'])