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
df = pd.read_csv("../input/train.csv")
df = df[['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'LotArea', 'LotFrontage', 'SalePrice', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',  'GarageArea', 'WoodDeckSF', 'OpenPorchSF',  'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']]


print(df.head())
import numpy as np
from sklearn import preprocessing,cross_validation,svm
from sklearn.linear_model import LinearRegression
df.dropna(inplace=True)
X=np.array(df.drop(['SalePrice'],1))
Y=np.array(df['SalePrice'])
X = preprocessing.scale(X)
X_train,X_test,y_train,y_test = cross_validation.train_test_split(X,Y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)
accuracy = clf.score(X_test,y_test)
#df2= pd.read_csv("../input/test.csv")
#df2 = df2[['TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', 'BsmtFinSF2', 'LotArea', 'LotFrontage', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'GarageCars',  'GarageArea', 'WoodDeckSF', 'OpenPorchSF',  'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea']]
##df2.dropna(inplace=True)
#######ccuracy_TEST = clf.score(X_TEST,Y_TEST)
print(accuracy)