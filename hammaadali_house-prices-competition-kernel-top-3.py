# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
train = pd.read_csv('../input/home-data-for-ml-course/train.csv')
test = pd.read_csv('../input/home-data-for-ml-course/test.csv')
train.shape, test.shape
trt = train.copy()
print(trt.shape)
trt = pd.get_dummies(trt, drop_first=True)
trt.shape
cols = trt.columns[1:15]    # change indexes to plot different features against SalePrice
fig, ax = plt.subplots(len(cols), figsize=(8,55))
fig.subplots_adjust(hspace=1)
for i, c in enumerate(cols):
    ax[i].scatter(trt[c], trt['SalePrice'])
    ax[i].set_yticks(range(0, trt['SalePrice'].max(), 100_000))
    ax[i].grid()
    ax[i].set_title(c)
plt.show()
outliers = []

# outliers.append(trt[trt['OverallCond']==5][trt['SalePrice']>700_000].index)
# outliers.append(trt[trt['OverallCond']==2][trt['SalePrice']>300_000].index)
# outliers.append(trt[trt['OverallCond']==6][trt['SalePrice']>700_000].index)
outliers.append(trt[trt['OverallQual']==10][trt['SalePrice']<200_000].index)
outliers.append(trt[trt['LotArea']>100_000].index)
outliers.append(trt[trt['LotFrontage']>300].index)
outliers.append(trt[trt['YearBuilt']<1900][trt['SalePrice']>200_000].index)
outliers.append(trt[trt['YearRemodAdd']<2000][trt['SalePrice']>600_000].index)
outliers.append(trt[trt['MasVnrArea']==1600].index)
outliers.append(trt[trt['TotalBsmtSF']>3000][trt['SalePrice']<300_000].index)
outliers.append(trt[trt['1stFlrSF']>2700][trt['SalePrice']<500_000].index)
outliers.append(trt[trt['BsmtFullBath']==3.0].index)
outliers.append(trt[trt['GrLivArea']>3300][trt['SalePrice']<300_000].index)
outliers.append(trt[trt['FullBath']==0.0][trt['SalePrice']>300_000].index)
outliers.append(trt[trt['GarageArea']>1200][trt['SalePrice']<200_000].index)
outliers.append(trt[trt['OpenPorchSF']>500].index)

outliers = [x[0] for x in outliers]
outliers
train.drop(outliers, axis=0, inplace=True)
y = train['SalePrice']
train.head(1)
trt.shape, train.shape
df = pd.concat([train.drop(['SalePrice'], axis=1), test], join='outer')
df.drop(['Id'], axis=1)
df.shape
empty = [x for x in df if df[x].isna().sum() != 0]
for x in empty:
    if df[x].dtype == 'float64':
        print(x)
    if df[x].dtype == 'int64':
        print(x)
to_mode = [
    'MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'Electrical', 'KitchenQual', 'Functional',
    'SaleType', 'LotFrontage'
]

to_none = [
    'Alley', 'MasVnrType', 'BsmtQual', 'BsmtExposure', 'BsmtCond', 'BsmtFinType1', 'BsmtFinType2', 
    'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence',
    'MiscFeature'
]

to_mean = [
    'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath',
    'BsmtHalfBath', 'GarageYrBlt', 'GarageCars', 'GarageArea'
]
for x in to_mod:
    df[x+'for_na'] = df[x].apply(lambda a:0 if pd.isnull(a)==True else 1)
    df[x] = df[x].fillna(df[x].mode()[0])
for x in to_zero:
    df[x+'for_na'] = df[x].apply(lambda a:0 if pd.isnull(a)==True else 1)
    df[x] = df[x].fillna(df[x].mean())
for x in to_mean:
    df[x+'for_na'] = df[x].apply(lambda a:0 if pd.isnull(a)==True else 1)
    df[x] = df[x].fillna('None')
df.isna().sum()
train.corr()[-1:]
cols_to_drop = [ 'YrSold', 'MoSold', 'BsmtHalfBath', 'BsmtFinSF2', 'KitchenAbvGr',
                'LowQualFinSF', 'BedroomAbvGr', '3SsnPorch', 
               ]
df.drop(cols_to_drop, axis=1, inplace=True)
df['MSSubClass'] = df['MSSubClass'].astype('category')
df = pd.get_dummies(df)
print(df.isna().sum().sum())
df.head(1)
X = df[:train.shape[0]]
train = df[:train.shape[0]]
tst = df[train.shape[0]:]
X.shape, tst.shape
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
X = StandardScaler().fit_transform(X)
tst = StandardScaler().fit_transform(tst)
X.shape
X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.01)
#Ridge
params = {
    'alpha': [25, 35],
    'max_iter': [None, 1000, 5000],
    'solver': ['svd', 'lsqr', 'sag', 'saga', 'sparse_cg', 'sparse_cg']
}

M1 = GridSearchCV(
    Ridge(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)
M1.fit(X, y)

print(M1.best_estimator_)

mean_absolute_error(y_ts, M1.predict(X_ts))
# Lasso
params = {
    'alpha': [0.1, 1, 3],
    'max_iter': [40_000],
}

M2 = GridSearchCV(
    Lasso(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)

M2.fit(X, y)
print(M2.best_estimator_)

mean_absolute_error(y_ts, M2.predict(X_ts))
# SVC
params = {
    'kernel': ['rbf', 'sigmoid', 'linear'],
    'C'  : [0,0.5,1,4],
    'gamma' : [None, 0.01, 0.1, 1, 3]  
}

M4 = GridSearchCV(
    SVR(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
)

M4.fit(X, y)
print(M4.best_estimator_)

mean_absolute_error(y_ts, M4.predict(X_ts))
#Gradient Boost
params = {
    'n_estimators': [500],
    'learning_rate': [0.01, 0.03, 0.1, 1],
    'loss': ['ls'],
}

M5 = GridSearchCV(
    GradientBoostingRegressor(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
).fit(X,y)

print(M5.best_estimator_)

mean_absolute_error(y_ts, M5.predict(X_ts))
# XG boost
params = {
    'learning_rate': [0.003, 0.01],
    'n_estimators': [3000, 4000],
    'max_depth': [2, 3],
    'min_child_weight': [0, 1],
    'gamma': [0],
    'subsample': [0.5, 0.7],
    'colsample_bytree':[0.5, 0.7],
    'objective': ['reg:squarederror'],
    'scale_pos_weight': [1, 2],
    'reg_alpha': [0.00001, 0.001]
}

M6 = GridSearchCV(
    XGBRegressor(),
    scoring='neg_mean_absolute_error',
    param_grid=params,
    cv=3,
    n_jobs=-1,
    verbose=1
).fit(X,y)

print(M6.best_estimator_)

mean_absolute_error(y_ts, M6.predict(X_ts))
# model = XGBRegressor(learning_rate=0.01, n_estimators=3460,
#                      max_depth=3, min_child_weight=0,
#                      gamma=0, subsample=0.7,
#                      colsample_bytree=0.7,
#                      objective='reg:squarederror', nthread=-1,
#                      scale_pos_weight=1, seed=27,
#                      reg_alpha=0.00006)
# model.fit(X,y)
# mean_absolute_error(y_ts, model.predict(X_ts))
preds = M6.predict(tst)
tst.shape
submit_file = pd.read_csv('../input/home-data-for-ml-course/sample_submission.csv')
submit_file['SalePrice'] = preds
submit_file.to_csv('submission.csv', index=False)
submit_file.shape