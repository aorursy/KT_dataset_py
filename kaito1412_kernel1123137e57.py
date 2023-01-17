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
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')
train.describe
train.corr()
import matplotlib.pyplot as plt

import seaborn as sns

sns.countplot(x = 'SalePrice', hue = "FullBath", data = train)
train.isnull().sum()
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
train["LotFrontage"].fillna(train["LotFrontage"].median(skipna=True), inplace=True)
train.isnull().sum()
train.isnull().sum()[train.isnull().sum()>0].sort_values(ascending=False)
from sklearn.preprocessing import StandardScaler

train_standard = StandardScaler()

train_copied = train.copy()

train_standard.fit(train_copied[['YearBuilt']])

train_std = pd.DataFrame(train_standard.transform(train_copied[['YearBuilt']]))

train_std

train[['YearBuilt'] ] = train_std

train
train.drop(['Id','MSSubClass','MSZoning','Street','Alley','LotShape','LandContour','Utilities','LotConfig','LandSlope','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','OverallQual','OverallCond','YearRemodAdd','RoofStyle','RoofMatl','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','ExterCond','Foundation','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtFinType2','BsmtFinSF2','BsmtUnfSF','TotalBsmtSF','Heating','HeatingQC','CentralAir','Electrical','1stFlrSF','2ndFlrSF','LowQualFinSF','GrLivArea','BsmtFullBath','BsmtHalfBath','HalfBath','BedroomAbvGr','KitchenAbvGr','KitchenQual','TotRmsAbvGrd','Functional','Fireplaces','FireplaceQu','GarageType','GarageYrBlt','GarageFinish','GarageCars','GarageArea','GarageQual','GarageCond','PavedDrive','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea','PoolQC','Fence','MiscFeature','MiscVal','MoSold','YrSold','SaleType','SaleCondition', ], axis=1, inplace=True)
from sklearn.linear_model import LogisticRegression

cols = ["LotFrontage","LotArea","YearBuilt","FullBath"] 

X = train[cols]

y = train['SalePrice']

# Build a logreg and compute the feature importances

model = LogisticRegression()

# create the RFE model and select 8 attributes

model.fit(X,y)

from sklearn.metrics import accuracy_score

train_predicted = model.predict(X)

accuracy_score(train_predicted, y)
train