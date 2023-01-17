# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


train_data = pd.read_csv('../input/train.csv')
test_data = pd.read_csv('../input/test.csv')

numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
Y = train_data['SalePrice']
X = train_data[['EnclosedPorch', 'KitchenAbvGr', 'LotArea' ,'OverallQual', 'BsmtFinSF1' ,'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','BedroomAbvGr','TotRmsAbvGrd','GarageArea','WoodDeckSF','ScreenPorch']]
from sklearn.ensemble import ExtraTreesRegressor
clf = ExtraTreesRegressor()
clf.fit(X,Y)
print(clf.feature_importances_)
X_train, X_test, y_train, y_test = train_test_split(X,Y,random_state= 0)
reg = RandomForestRegressor().fit(X,Y)
test_fin = test_data[['EnclosedPorch', 'KitchenAbvGr', 'LotArea' ,'OverallQual', 'BsmtFinSF1' ,'BsmtUnfSF', 'TotalBsmtSF','1stFlrSF','2ndFlrSF','GrLivArea','FullBath','BedroomAbvGr','TotRmsAbvGrd','GarageArea','WoodDeckSF','ScreenPorch']]
test_fin['GarageArea'].fillna(test_fin['GarageArea'].median())
test_fin['BsmtFinSF1'].fillna(test_fin['BsmtFinSF1'].median())
test_fin['BsmtUnfSF'].fillna(test_fin['BsmtUnfSF'].median())
test_fin['TotalBsmtSF'].fillna(test_fin['TotalBsmtSF'].median())

test_fin.info()
test_fin.BsmtFinSF1.fillna(test_fin.BsmtFinSF1.median(),inplace = True)
test_fin.BsmtUnfSF.fillna(test_fin.BsmtUnfSF.median(),inplace = True)
test_fin.TotalBsmtSF.fillna(test_fin.TotalBsmtSF.median(),inplace = True)
test_fin.GarageArea.fillna(test_fin.GarageArea.median(),inplace = True)

#test_fin.drop(['newBsmtFinSF1'])
test_fin.info()
val = reg.predict(test_fin)
s = pd.DataFrame()
s['Id'] = s.index
s['SalePrice'] = val
s['Id'] = s.index + 1461
s.to_csv('a_out.csv', index = False)
s
