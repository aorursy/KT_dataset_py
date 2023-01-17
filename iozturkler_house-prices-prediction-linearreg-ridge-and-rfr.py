# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np

import pandas as pd

from scipy.stats import skew

from sklearn.ensemble import RandomForestRegressor

from sklearn.linear_model import Ridge

from sklearn.linear_model import LinearRegression

from mlxtend.regressor import StackingRegressor

from sklearn.metrics import mean_squared_error

from math import sqrt

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_train = pd.read_csv("../input/train.csv")

df_test = pd.read_csv("../input/test.csv")

print (df_train.shape)

print (df_test.shape)
df_train.head()
# Aykırı olanların kaldırılması

df_train = df_train.drop(

    df_train[(df_train['GrLivArea']>4000) & (df_train['SalePrice']<300000)].index)
# dataların birleştirilmesi

all_data = pd.concat((df_train.loc[:,'MSSubClass':'SaleCondition'],

                      df_test.loc[:,'MSSubClass':'SaleCondition']))
# utilities sütünunun kaldırılması

all_data = all_data.drop(['Utilities'], axis=1)
# Eksik değerlerin doldurulması

all_data["PoolQC"] = all_data["PoolQC"].fillna("None")

all_data["MiscFeature"] = all_data["MiscFeature"].fillna("None")

all_data["Alley"] = all_data["Alley"].fillna("None")

all_data["Fence"] = all_data["Fence"].fillna("None")

all_data["FireplaceQu"] = all_data["FireplaceQu"].fillna("None")

for col in ('GarageType', 'GarageFinish', 'GarageQual', 'GarageCond'):

    all_data[col] = all_data[col].fillna('None')

for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')

all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])

all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["Functional"] = all_data["Functional"].fillna("Typ")

all_data['Electrical'] = all_data['Electrical'].fillna(all_data['Electrical'].mode()[0])

all_data['KitchenQual'] = all_data['KitchenQual'].fillna(all_data['KitchenQual'].mode()[0])

all_data['Exterior1st'] = all_data['Exterior1st'].fillna(all_data['Exterior1st'].mode()[0])

all_data['Exterior2nd'] = all_data['Exterior2nd'].fillna(all_data['Exterior2nd'].mode()[0])

all_data['SaleType'] = all_data['SaleType'].fillna(all_data['SaleType'].mode()[0])

all_data['MSSubClass'] = all_data['MSSubClass'].fillna("None")

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)

    

for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)

    

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
#Sayısal değerleri kategorik değerlere dönüştürme

all_data['MSSubClass'] = all_data['MSSubClass'].apply(str)

all_data['OverallCond'] = all_data['OverallCond'].astype(str)

all_data['YrSold'] = all_data['YrSold'].astype(str)

all_data['MoSold'] = all_data['MoSold'].astype(str)
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index

skewed_feats = all_data[numeric_feats].apply(lambda x: skew(x.dropna())) #compute skewness

skewed_feats = skewed_feats[skewed_feats > 0.75]

skewed_feats = skewed_feats.index

all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
#train,test ve validation datalarının ayrılması

X_train = all_data[:df_train.shape[0]]

X_test = all_data[df_train.shape[0]:]

y = df_train.SalePrice
# modelin ayarlanması

linearrg = LinearRegression(n_jobs = -1)

rid = Ridge(alpha = 5)

randomfor = RandomForestRegressor(n_estimators = 12,max_depth = 3,n_jobs = -1)
model = StackingRegressor(regressors=[randomfor, rid],meta_regressor=linearrg)

model.fit(X_train, y)
# train datası üstünde tahminleme yapılması

y_pred = model.predict(X_train)

print(sqrt(mean_squared_error(y, y_pred)))
# test datasını tahmin et

Y_pred = model.predict(X_test)
sub = pd.DataFrame()

sub['Id'] = df_test['Id']

sub['SalePrice'] = np.expm1(Y_pred)

sub.to_csv('submission.csv',index=False)