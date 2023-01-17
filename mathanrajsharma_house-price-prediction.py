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
df_train = pd.read_csv('../input/train.csv')

df_test = pd.read_csv('../input/test.csv')
df_train.describe()
df_train.columns
df_train.dtypes
import matplotlib.pyplot as plt

import seaborn as sns

from scipy.stats import norm

from sklearn.preprocessing import StandardScaler

from scipy import stats

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline
#lets plot the distribution of Sales price

sns.distplot(df_train['SalePrice'])
print('Skewness: ', df_train['SalePrice'].skew())

print('Kurtosis: ', df_train['SalePrice'].kurt())
df_train['Age'] = 2019-df_train['YearBuilt']

df_train['Remod_Age'] = 2019-df_train['YearRemodAdd']

df_train['Age'].describe()



#do the same for the testing set

df_test['Age'] = 2019-df_test['YearBuilt']

df_test['Remod_Age'] = 2019-df_test['YearRemodAdd']
df_train['Age'].head()
#lets plot the Age VS SalePrice Graph, Since now they have linear relationship lets plot a scatter plot

var = 'Age'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
var = 'Remod_Age'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
#drop the YearBuilt and YearRemodAdd columns from training and testing sets

df_train.drop(['YearBuilt','YearRemodAdd'], axis = 1)
df_test.drop(['YearBuilt','YearRemodAdd'], axis = 1)
#correlation matrix

corrmat = df_train.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
var = 'OverallQual'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'GarageCars'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'FullBath'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'TotRmsAbvGrd'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'Fireplaces'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
var = 'KitchenAbvGr'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)

f, ax = plt.subplots(figsize=(10, 6))

fig = sns.boxplot(x=var, y="SalePrice", data=data)

fig.axis(ymin=0, ymax=800000);
#lets calculate the Garage_Age

df_train['Garage_Age'] = 2019 - df_train['GarageYrBlt']

df_test['Garage_Age'] = 2019 - df_test['GarageYrBlt']

df_train['Garage_Age'].dtype
var = 'Garage_Age'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
var = 'MasVnrArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
var = '1stFlrSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
var = 'GrLivArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
var = 'GarageArea'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
var = 'EnclosedPorch'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
df_train = df_train[['GarageArea', 'GarageCars', 'Fireplaces', 'TotRmsAbvGrd', 'FullBath', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF', 'MasVnrArea', 'OverallQual','Age','Remod_Age', 'KitchenAbvGr', 'EnclosedPorch','Garage_Age','SalePrice']].copy()
df_test = df_test[['GarageArea', 'GarageCars', 'Fireplaces', 'TotRmsAbvGrd', 'FullBath', 'GrLivArea', '1stFlrSF', 'TotalBsmtSF', 'MasVnrArea', 'OverallQual','Age','Remod_Age', 'KitchenAbvGr', 'EnclosedPorch','Garage_Age']].copy()
#lets check if there exist any missing value in out dataset

df_train.isna().any()
#Impute missing values in MasVnrArea using the mean of that variable

df_train['MasVnrArea'].fillna((df_train['MasVnrArea'].mean()), inplace = True)
#Impute missing values in Garage_Age using the mode of that variable

df_train['Garage_Age'].fillna((df_train['Garage_Age'].mode()[0]), inplace = True)
df_train.isna().any()
#do the same for the testing data set

df_test.isna().any()
df_test['GarageArea'].fillna((df_test['GarageArea'].mean()), inplace = True)

df_test['GarageCars'].fillna((df_test['GarageCars'].mode()[0]), inplace = True)

df_test['TotalBsmtSF'].fillna((df_test['TotalBsmtSF'].mean()), inplace = True)

df_test['MasVnrArea'].fillna((df_test['MasVnrArea'].mean()), inplace = True)

df_test['Garage_Age'].fillna((df_test['Garage_Age'].mode()[0]), inplace = True)
#Check whether successfully imputed or not

df_test.isna().any()
X = df_train.drop(['SalePrice'], axis = 1)

y = df_train['SalePrice']
X = pd.get_dummies(X)

df_test = pd.get_dummies(df_test)
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression



X_train, X_eval, y_train, y_eval = train_test_split(X,y,test_size = 0.2)
model = LinearRegression()

model.fit(X_train,y_train)
from sklearn.metrics import mean_squared_error

from math import sqrt

pred_eval = model.predict(X_eval)

rmse = sqrt(mean_squared_error(y_eval,pred_eval))



print('RMSE: ', rmse)
def find_outliers_keys(x):

    q1 = np.percentile(x, 25)

    q3 = np.percentile(x, 75)

    iqr = q3-q1   

    floor = q1 - 1.5*iqr

    ceiling = q3 + 1.5*iqr

    outlier_indices = list(x.index[(x < floor)|(x > ceiling)])

    outlier_values = list(x[outlier_indices])



    return outlier_indices

out = find_outliers_keys(df_train['TotalBsmtSF'])
df_train = df_train.drop(out)
var = 'TotalBsmtSF'

data = pd.concat([df_train['SalePrice'], df_train[var]], axis = 1)

data.plot.scatter(x = var, y = 'SalePrice', ylim = (0,800000))
#lets rebuild the model

X = df_train.drop(['SalePrice'], axis = 1)

y = df_train['SalePrice']

X = pd.get_dummies(X)

X_train, X_eval, y_train, y_eval = train_test_split(X,y,test_size = 0.2)



model2 = LinearRegression()

model2.fit(X_train,y_train)



pred_eval = model2.predict(X_eval)

rmse = sqrt(mean_squared_error(y_eval,pred_eval))



print('RMSE: ', rmse)
from sklearn.ensemble import GradientBoostingRegressor

gb_reg = GradientBoostingRegressor(n_estimators=2000, learning_rate=0.02,

                                   max_depth=4, max_features='sqrt',

                                   min_samples_leaf=30, min_samples_split=30, 

                                   loss='huber')

gb_reg.fit(X_train, y_train)

pred_eval = gb_reg.predict(X_eval)

rmse = sqrt(mean_squared_error(y_eval,pred_eval))



print('RMSE: ', rmse)
from sklearn.ensemble import RandomForestRegressor



forest_reg = RandomForestRegressor(n_estimators=1000,max_features=14)

forest_reg.fit(X_train,y_train);

pred_eval = forest_reg.predict(X_eval)

rmse = sqrt(mean_squared_error(y_eval,pred_eval))



print('RMSE: ', rmse)
import xgboost as xgb



model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 

                             learning_rate=0.05, max_depth=3, 

                             min_child_weight=1.7817, n_estimators=2200,

                             reg_alpha=0.4640, reg_lambda=0.8571,

                             subsample=0.5213, silent=1,

                             random_state =7, nthread = -1)

model_xgb.fit(X_train,y_train)

pred_eval = model_xgb.predict(X_eval)

rmse = sqrt(mean_squared_error(y_eval,pred_eval))



print('RMSE: ', rmse)