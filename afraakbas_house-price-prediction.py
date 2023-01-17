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
import math
import matplotlib.pyplot as plt
import seaborn as sns
import csv
submission = open('/kaggle/working/my_submission.csv', 'w')
    
color = sns.color_palette()
sns.set_style('darkgrid')
train = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/train.csv")
test = pd.read_csv("/kaggle/input/house-prices-advanced-regression-techniques/test.csv")
#solutions = pd.read_csv("sample_submission.csv")
train.drop("Id", axis=1, inplace=True)
test.drop("Id", axis=1, inplace=True)
#a = solutions.Id.values
#solutions.drop("Id", axis=1, inplace=True)
print("train shape ", train.shape, " test shape ", test.shape)
y_train = train.SalePrice.values
correlation = train.corr()
plt.subplots(figsize=(12,9))
sns.heatmap(correlation, vmax=0.9)
plt.show()
large_corr = correlation[correlation['SalePrice'] > 0.6]
small_corr = correlation[correlation['SalePrice'] < 0.05]
pd.DataFrame(large_corr)
pd.DataFrame(small_corr)
y_train = np.log1p(y_train)
null_data = (train.isnull().sum() / len(train)) * 100
null_data = null_data.drop(null_data[null_data == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :null_data})
missing_data
train = train.drop(missing_data[missing_data['Missing Ratio'] > 80].index,1)

for i in ['FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 
         'BsmtExposure', 'BsmtCond', 'BsmtQual', 'MasVnrType']:
    train[i] = train[i].fillna("None")

train['Exterior1st'] = train['Exterior1st'].fillna("Other")
train['Exterior2nd'] = train['Exterior2nd'].fillna("Other")
train['SaleType'] = train['SaleType'].fillna("Oth")
train['Utilities'] = train['Utilities'].fillna("AllPub")
train['Functional'] = train['Functional'].fillna("Typ")
train['Electrical'] = train['Electrical'].fillna("SBrkr")
from scipy import stats
from sklearn import impute 
from sklearn import utils
from sklearn import ensemble
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from scipy.stats import skew
from sklearn import metrics
from scipy.special import boxcox1p
from sklearn import tree
from sklearn.model_selection import KFold
imp = impute.SimpleImputer(strategy='most_frequent')
imp2 = impute.SimpleImputer(strategy='mean')

for j in ['BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt']:
    train[j] = imp.fit_transform(train[[j, 'SalePrice']])
    
for k in ['MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 
          'TotalBsmtSF', 'LotFrontage']:
    train[k] = imp2.fit_transform(train[[k, 'SalePrice']])
    
for l in ['GarageCars', 'MasVnrArea', 'BsmtUnfSF', 'BsmtFinSF2', 
          'BsmtFinSF1', 'GarageArea', 'GarageYrBlt']:
    train[l] = train[l].fillna(0)
    
for m in['MSZoning', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2']:
    train[m] = train[m].fillna(train[m].mode()[0])
null_data = (test.isnull().sum() / len(test)) * 100
null_data = null_data.drop(null_data[null_data == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :null_data})
missing_data
test = test.drop(missing_data[missing_data['Missing Ratio'] > 80].index,1)

for i in ['FireplaceQu', 'GarageCond', 'GarageFinish', 'GarageQual', 'GarageType', 
         'BsmtExposure', 'BsmtCond', 'BsmtQual', 'MasVnrType']:
    test[i] = test[i].fillna("None")

test['Exterior1st'] = test['Exterior1st'].fillna("Other")
test['Exterior2nd'] = test['Exterior2nd'].fillna("Other")
test['SaleType'] = test['SaleType'].fillna("Oth")
test['Utilities'] = test['Utilities'].fillna("AllPub")
test['Functional'] = test['Functional'].fillna("Typ")
test['Electrical'] = test['Electrical'].fillna("SBrkr")
for l in ['GarageCars', 'MasVnrArea', 'BsmtUnfSF', 'BsmtFinSF2', 
          'BsmtFinSF1', 'GarageArea', 'GarageYrBlt']:
    test[l] = test[l].fillna(0)
    
for m in['MSZoning', 'KitchenQual', 'BsmtFinType1', 'BsmtFinType2']:
    test[m] = test[m].fillna(test[m].mode()[0])
    
for j in ['BsmtFullBath', 'BsmtHalfBath', 'GarageYrBlt']:
    test[j] = imp.fit_transform(test[[j]])
    
for k in ['MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 
          'TotalBsmtSF', 'LotFrontage']:
    test[k] = imp2.fit_transform(test[[k]])
print("train shape ", train.shape, " test shape ", test.shape)
from sklearn import preprocessing
lbl = preprocessing.LabelEncoder()
col = train.select_dtypes(include=['object'])
for labels in col:
    train[labels] = lbl.fit_transform(train[labels])
col = test.select_dtypes(include=['object'])
for labels in col:
    test[labels] = lbl.fit_transform(test[labels])
for c in large_corr.transpose().columns:
    plt.figure(figsize=(5,4))
    plt.scatter(train[c], train['SalePrice'])
    plt.xlabel(c)
    plt.show()
train = train.drop(train[train['TotalBsmtSF'] > 6000].index)

train = train.drop(train[(train['GrLivArea'] > 4000) & (train['SalePrice'] < 200000)].index)
train = train.drop(train[(train['BsmtFinSF2'] > 1400) &  (train['SalePrice'] < 350000)].index)

train = train.drop(train[(train['EnclosedPorch'] > 500) & (train['SalePrice'] < 300000)].index)

train = train.drop(train[(train['MiscVal'] > 8000) & (train['SalePrice'] < 200000)].index)
y_train = train.SalePrice.values
d_tr = tree.DecisionTreeRegressor()
gr_b = ensemble.GradientBoostingRegressor(loss='ls', learning_rate=0.05, n_estimators=300)
cv = KFold(n_splits=5,shuffle=False)
r2 = metrics.make_scorer(metrics.mean_squared_error)
train.drop("SalePrice", axis=1, inplace=True)
train.shape
d_tr.fit(train, y_train)
d_tr_pre = d_tr.predict(test)
print("r2 score ", metrics.r2_score(y_train, d_tr_pre))
print("cross_val_score ", cross_val_score(d_tr, train, d_tr_pre, cv=cv).mean())
np.sqrt(metrics.mean_squared_error(y_train, d_tr_pre))
d_tr_pre[:10], y_train[:10]
gr_b.fit(train, y_train)
gr_pre = gr_b.predict(train)
print("r2 score ", metrics.r2_score(y_train, gr_pre))
print("cross_val_score ", cross_val_score(d_tr, train, gr_pre, cv=cv).mean())
np.sqrt(metrics.mean_squared_error(y_train, gr_pre))
d_tr_pre[:10], y_train[:10]
