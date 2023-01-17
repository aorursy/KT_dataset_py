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
import numpy as np

import pandas as pd

import seaborn as sns

from heapq import nlargest 

import scipy.stats as stats

import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec

import seaborn as sns; sns.set()

from sklearn.preprocessing import LabelEncoder

from sklearn.compose import ColumnTransformer

from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split

from xgboost import XGBRegressor

import lightgbm as lgb

from sklearn.linear_model import LinearRegression, Ridge, RidgeCV

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import cross_val_score

import warnings

warnings.filterwarnings('ignore')

pd.set_option('max_columns', 105)
train = pd.read_csv("../input/house-prices-advanced-regression-techniques/train.csv")

test = pd.read_csv("../input/house-prices-advanced-regression-techniques/test.csv")
train.columns
train.head()
train['SalePrice'].describe()
train.describe()
plt.xticks(rotation=90)

sns.distplot(train['SalePrice'], fit=stats.norm)
#skewness and kurtosis

print("Skewness: %f" % train['SalePrice'].skew())

print("Kurtosis: %f" % train['SalePrice'].kurt())
train.plot.scatter(x='GrLivArea', y='SalePrice');

# 'TotalBsmtSF' -- highlight zero values
plt.subplots(figsize=(10, 8))

sns.boxplot(x='OverallQual', y="SalePrice", data=train)
plt.subplots(figsize=(14, 8))

plt.xticks(rotation=90)

sns.boxplot(x='YearBuilt', y="SalePrice", data=train)
plt.subplots(figsize=(14, 8))

plt.xticks(rotation=90)

sns.boxplot(x='Neighborhood', y="SalePrice", data=train)
#correlation matrix

corrmat = train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat)
#saleprice correlation matrix

k = 10 #number of variables for heatmap

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(train[cols].values.T)

f, ax = plt.subplots(figsize=(12, 9))

#sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
numerical_feats = train.dtypes[train.dtypes != "object"].index

print("Number of Numerical features: ", len(numerical_feats))



categorical_feats = train.dtypes[train.dtypes == "object"].index

print("Number of Categorical features: ", len(categorical_feats))
print(train[numerical_feats].columns)

print("*"*100)

print(train[categorical_feats].columns)

train[numerical_feats].head()
train[categorical_feats].head()
train.plot.scatter(x='GrLivArea', y='SalePrice');
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<250000)].index).reset_index(drop=True)
train.plot.scatter(x='GrLivArea', y='SalePrice');
y_train = train['SalePrice']

all_data = pd.concat([train, test]).reset_index(drop=True)

all_data.drop(['SalePrice'], axis=1, inplace=True)

all_data.shape
total = all_data.isnull().sum().sort_values(ascending=False)

percent = (all_data.isnull().sum()/all_data.isnull().count()).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1)

missing_data.head(20)
drop_missing = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu']



for col in drop_missing:

    all_data.drop([col], axis=1, inplace=True)



all_data.shape
fill_missing = ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']



for col in fill_missing:

    all_data[col] = all_data[col].fillna('None')
for col in ('GarageYrBlt', 'GarageArea', 'GarageCars'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):

    all_data[col] = all_data[col].fillna(0)
for col in ('BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):

    all_data[col] = all_data[col].fillna('None')
all_data["MasVnrType"] = all_data["MasVnrType"].fillna("None")

all_data["MasVnrArea"] = all_data["MasVnrArea"].fillna(0)
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood

all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(

    lambda x: x.fillna(x.median()))
all_data['MSZoning'].value_counts()
all_data['MSZoning'] = all_data['MSZoning'].fillna(all_data['MSZoning'].mode()[0])
all_data['Utilities'].value_counts()
all_data["Functional"].value_counts()
all_data["Functional"] = all_data["Functional"].fillna("Typ")
all_data = all_data.drop(['Utilities'], axis=1)
mode_col = ['Electrical','KitchenQual', 'Exterior1st', 'Exterior2nd', 'SaleType']

for col in mode_col:

    all_data[col] = all_data[col].fillna(all_data[col].mode()[0])
all_data['MSSubClass'].value_counts()
all_data = all_data.drop(['Id'], axis=1)
all_data.columns
all_data = pd.get_dummies(all_data)
all_data.shape
train['SalePrice_log'] = np.log(train['SalePrice'])
sns.distplot(train['SalePrice_log'], fit=stats.norm);
# Splitting data

y = train['SalePrice_log']

X = all_data[:train.shape[0]]

X_sub = all_data[train.shape[0]:]
X_train, X_valid, y_train, y_valid = train_test_split(X,y, test_size=0.2)
lm = LinearRegression()

lm.fit(X_train, y_train)
lm.score(X_train,y_train)
y_pred = lm.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_pred, y_valid))

rmse
rdgCV = RidgeCV(alphas=[0.01,0.1,1,10,100,1000], cv=5)

rdgCV.fit(X_train,y_train)
print(rdgCV.alpha_)
rdg = Ridge(alpha=10)

rdg.fit(X_train, y_train)

rdg.score(X_valid, y_valid)
y_pred = rdg.predict(X_valid)

rmse = np.sqrt(mean_squared_error(y_pred, y_valid))

rmse
submission_predictions = np.exp(rdg.predict(X_sub))
submission = pd.DataFrame({

        "Id": test["Id"],

        "SalePrice": submission_predictions

    })
submission.to_csv("ridge_predict.csv", index=False)