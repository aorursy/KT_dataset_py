# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')
df_train.head(10)
df_train.info()
df_train.describe()
df_train['LotFrontage'] = df_train['LotFrontage'].fillna(df_train['LotFrontage'].mean())
df_train['MasVnrArea'] = df_train['MasVnrArea'].fillna(df_train['MasVnrArea'].mean())
df_train['GarageYrBlt'] = df_train['GarageYrBlt'].fillna(df_train['GarageYrBlt'].mean())
df_test['LotFrontage'] = df_test['LotFrontage'].fillna(df_test['LotFrontage'].mean())
df_test['MasVnrArea'] = df_test['MasVnrArea'].fillna(df_test['MasVnrArea'].mean())
df_test['GarageYrBlt'] = df_test['GarageYrBlt'].fillna(df_test['GarageYrBlt'].mean())
# Repartition of the sale price

sns.distplot(df_train['SalePrice']);
sns.distplot(np.log1p(df_train['SalePrice']));
cor_mat = df_train[:].corr()
mask = np.array(cor_mat)
mask[np.tril_indices_from(mask)] = False
fig = plt.gcf()
fig.set_size_inches(80,15)
sns.heatmap(data=cor_mat, mask=mask, square=True, annot=True, cbar=True);
var = 'OverallQual'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
sns.boxplot(x=var, y="SalePrice", data=data);
var = 'GrLivArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.xlabel('GrLivArea')
plt.ylabel('SalePrice')
plt.scatter(x=var, y="SalePrice", data=data);
df_train = df_train[df_train['GrLivArea'] < 4000]
var = 'GarageCars'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
sns.boxplot(x=var, y="SalePrice", data=data);
var = 'GarageArea'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
plt.xlabel('GarageArea')
plt.ylabel('SalePrice')
plt.scatter(x=var, y="SalePrice", data=data);
var = 'TotalBsmtSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice');
df_train = df_train[df_train['TotalBsmtSF'] < 3000]
var = '1stFlrSF'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice');
df_train = df_train[df_train['1stFlrSF'] < 3000]
X = df_train.drop(columns= ['SalePrice', 'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'])
X_test_true = df_test.drop(columns= [ 'Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Electrical', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature'])
y = np.log1p(df_train['SalePrice'])
X.shape, X_test_true.shape, y.shape
import xgboost as xgb

from sklearn.model_selection import train_test_split,KFold, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, RobustScaler
one_hot_encoded_training_predictors = pd.get_dummies(X)
one_hot_encoded_test_predictors = pd.get_dummies(X_test_true)
X, X_test_true = one_hot_encoded_training_predictors.align(one_hot_encoded_test_predictors, join='inner', axis=1)
X.shape, X_test_true.shape
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=9000)
X_train.shape, y_train.shape, X_test.shape, y_test.shape
kf = KFold(n_splits=5, random_state=42)
imputer = Imputer()
rb = RobustScaler()
xgbr = xgb.XGBRegressor()
pipe = Pipeline([('imputer', imputer), ('rb', rb), ('xgbr', xgbr)])
pipe.get_params().keys()
param_grid = {'imputer__strategy' : ['median', 'mean'],
              'xgbr__n_estimators' : [750, 1000],
              'xgbr__learning_rate' : [0.03, 0.04, 0.05]
               }
gs = GridSearchCV(pipe, param_grid=param_grid, cv=kf, verbose=3)
gs.fit(X_train, y_train)
gs.best_params_
best_model = gs.best_estimator_
best_model.fit(X_train, y_train)
y_predic = np.exp(best_model.predict(X_test))
mse_scores = -cross_val_score(best_model, X_train, y_train, cv=kf, scoring='neg_mean_squared_log_error')
rmse_scores = mse_scores**0.5
rmse_scores.mean(), rmse_scores.std()
predicted_prices = np.expm1(best_model.predict(X_test_true))
my_submission = pd.DataFrame({'Id': X_test_true.Id, 'SalePrice': predicted_prices})
my_submission.to_csv('submission.csv', index=False)
