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
import pandas_profiling as pp
import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns

plt.style.use('fivethirtyeight')

matplotlib.rcParams['font.family'] = "Arial"
import collections

import itertools



import scipy.stats as stats

from scipy.stats import norm

from scipy.special import boxcox1p



import statsmodels

import statsmodels.api as sm
from sklearn.preprocessing import scale, StandardScaler, RobustScaler, OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import xgboost as xgb

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.svm import SVR
df = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
pp.ProfileReport(df)
df['reviews_per_month'].fillna(0, inplace = True)
df.drop(['id', 'name', 'last_review', 'host_name'], axis = 1, inplace = True)
fig, axes = plt.subplots(1, 3, figsize = (21,6))

sns.distplot(df['price'], ax = axes[0])

sns.distplot(np.log1p(df['price']), ax = axes[1])

axes[1].set_xlabel('log(1 + price)')

sm.qqplot(np.log1p(df['price']), stats.norm, fit=True, line='45', ax=axes[2]);
df = df[np.log1p(df['price']) < 7]

df = df[np.log1p(df['price']) > 3]
fig, axes = plt.subplots(1, 3, figsize = (21,6))

sns.distplot(df['price'], ax = axes[0])

sns.distplot(np.log1p(df['price']), ax = axes[1])

axes[1].set_xlabel('log(1 + price)')

sm.qqplot(np.log1p(df['price']), stats.norm, fit=True, line='45', ax=axes[2]);
df['price'] = np.log1p(df['price'])
fig, axes = plt.subplots(1,2, figsize = (21,6))

sns.distplot(df['minimum_nights'],kde = False, ax = axes[0])

axes[0].set_yscale('log')

axes[0].set_xlabel('minimum stay [nights]')

axes[0].set_ylabel('count')



sns.distplot(np.log1p(df['minimum_nights']), kde = False, ax = axes[1])

axes[1].set_yscale('log')

axes[1].set_xlabel('minimum stay [nights]')

axes[1].set_ylabel('count')
df['minimum_nights'] = np.log1p(df['minimum_nights'])
fig , axes = plt.subplots(1,1, figsize = (21,6))

sns.scatterplot(x = df['availability_365'], y = df['reviews_per_month'])
df['reviews_per_month'] = df[df['reviews_per_month'] < 15]['reviews_per_month']
df['reviews_per_month'].fillna(0, inplace = True)
cat_feat = df.select_dtypes(include = ['object'])

cat_feat_one_hot = pd.get_dummies(cat_feat)
cat_feat_one_hot.head()
num_feat = df.select_dtypes(exclude = ['object'])

y = num_feat.price

num_feat = num_feat.drop(['price'], axis = 1)
y_df = pd.DataFrame(y)
X = np.concatenate((num_feat, cat_feat_one_hot), axis = 1)

X_df = pd.concat([num_feat, cat_feat_one_hot], axis = 1)
data = pd.concat([X_df, y], axis = 1)

data.to_csv('NYC_airbnb_preprocessed.dat')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 50)
scaler = RobustScaler()

X_train = scaler.fit_transform(X_train)

X_test = scaler.fit_transform(X_test)
y_train.shape
X_train_df = pd.DataFrame(X_train)

X_test_df = pd.DataFrame(X_test)

y_train_df = pd.DataFrame(y_train)

y_test_df = pd.DataFrame(y_test)
best_rf = RandomForestRegressor(n_estimators = 80, min_samples_split = 10, max_features = 'auto', max_depth = None, bootstrap = True)

best_rf.fit(X_train, y_train)

y_pred = best_rf.predict(X_test)

mae = mean_absolute_error(y_test, best_rf.predict(X_test))

mape = 100*mae

acc = 100 - mape

print(acc)
print(r2_score(y_test,y_pred))
print(np.sqrt(mean_squared_error(y_test,y_pred)))