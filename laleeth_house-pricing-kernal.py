# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import missingno as msn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV

# from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone

# from sklearn.linear_model import ElasticNet, Lasso,  BayesianRidge, LassoLarsIC

# from sklearn.ensemble import RandomForestRegressor,  GradientBoostingRegressor

# from sklearn.model_selection import KFold, cross_val_score, train_test_split

# from sklearn.metrics import make_scorer, accuracy_score

# from sklearn.ensemble import RandomForestClassifier

# from sklearn.model_selection import cross_val_score

# from sklearn.metrics import classification_report

# from sklearn.model_selection import GridSearchCV

# from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import mean_squared_error

# from sklearn.preprocessing import RobustScaler

# from sklearn.metrics import confusion_matrix

# from sklearn.kernel_ridge import KernelRidge

# from sklearn.pipeline import make_pipeline

# from sklearn.metrics import accuracy_score

# from sklearn.pipeline import Pipeline

# import matplotlib.pyplot as plt

from scipy.stats import skew

import scipy.stats as stats

import lightgbm as lgb

import seaborn as sns

import xgboost as xgb

import matplotlib

import warnings

import sklearn

import scipy

import json

import sys

import csv

import os
print('matplotlib: {}'.format(matplotlib.__version__))

print('sklearn: {}'.format(sklearn.__version__))

print('scipy: {}'.format(scipy.__version__))

print('seaborn: {}'.format(sns.__version__))

print('pandas: {}'.format(pd.__version__))

print('numpy: {}'.format(np.__version__))

print('Python: {}'.format(sys.version))
train = pd.read_csv('../input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('../input/house-prices-advanced-regression-techniques/test.csv')

train.head()

test.head()
corrmat = train.corr()

f, ax = plt.subplots(figsize=(25, 15))

sns.heatmap(corrmat, vmax=.8, annot=True,annot_kws={"size": 8});
corrmat = train.corr()

top_corr_features = corrmat.index[abs(corrmat["SalePrice"])>0.5]

plt.figure(figsize=(10,10))

g = sns.heatmap(train[top_corr_features].corr(),annot=True,cmap="coolwarm")
sns.barplot(train.OverallQual,train.SalePrice)
sns.barplot(train.GarageCars,train.SalePrice)
sns.barplot(train.FullBath,train.SalePrice)
print(train['SalePrice'].skew()) 

print(train['SalePrice'].kurt())
null_columns=train.columns[train.isnull().any()]

nullo = train[null_columns].isnull().sum()

nullo
train['LotFrontage'].corr(train['LotArea'])
basement_cols=['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','BsmtFinSF1','BsmtFinSF2']

train[basement_cols][train['BsmtQual'].isnull()==True]
for col in basement_cols:

    if 'FinSF'not in col:

        train[col] = train[col].fillna('None')
garage_cols=['GarageType','GarageQual','GarageCond','GarageYrBlt','GarageFinish','GarageCars','GarageArea']

train[garage_cols][train['GarageType'].isnull()==True]
for col in garage_cols:

    if train[col].dtype==np.object:

        train[col] = train[col].fillna('None')

    else:

        train[col] = train[col].fillna(0)
train["MiscFeature"] = train["MiscFeature"].fillna('None')

train["Fence"] = train["Fence"].fillna('None')

train["Alley"] = train["Alley"].fillna('None')

train["FireplaceQu"] = train["FireplaceQu"].fillna('None')

train["PoolQC"] = train["PoolQC"].fillna('None')

train["LotFrontage"] = train["LotFrontage"].fillna('None')

train["MasVnrType"] = train["MasVnrType"].fillna('None')

train["MasVnrArea"] = train["MasVnrArea"].fillna('None')

train["Electrical"] = train["Electrical"].fillna('None')

train[null_columns].isnull().sum()
plt.scatter(train["1stFlrSF"],train.SalePrice, color='yellow')

plt.title("Sale Price wrt 1st floor")

plt.ylabel('Sale Price (in dollars)')

plt.xlabel("1st Floor in square feet");
train['SalePriceSF'] = train['SalePrice']/train['GrLivArea']

plt.hist(train['SalePriceSF'], bins=15,color="gold")

plt.title("Sale Price per Square Foot")

plt.ylabel('Number of Sales')

plt.xlabel('Price per square feet');
print("$",train.SalePriceSF.mean())