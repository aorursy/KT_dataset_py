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
%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import pandas as pd
#データのロード

Dataset = pd.read_csv("../input/kickstarter-projects/ks-projects-201801.csv")



#データの表示

Dataset.head(50)
#データの大きさ

print(Dataset.shape)

#columns

print(Dataset.columns)

print(Dataset.dtypes)
Dataset.describe()
print(Dataset.isnull().sum(axis = 0))
#Achievement

import copy

df_reg = copy.deepcopy(Dataset)

Achievement  = pd.DataFrame(df_reg['usd_pledged_real'] / df_reg['usd_goal_real'], columns=['Achievement'])

Dataset = pd.concat([Dataset, Achievement], axis=1)

Dataset.head()
#使用するデータを選定

df = Dataset[['category', 'main_category', 'currency', 'country', 'deadline', 'launched', 'goal', 'usd_goal_real', 'usd_pledged_real', 'Achievement' ]]

df.head(10)
import seaborn as sns

plt.figure(figsize = (15, 10))

scatter = plt.scatter(x = 'currency', y = 'Achievement', data = df)

#scatter.set(ylim = (0, 200))

plt.show()
plt.figure(figsize = (15, 10))

scatter = plt.scatter(x = 'country', y = 'Achievement', data = df)

#scatter.set(ylim = (0, 200))

plt.show()
plt.figure(figsize = (15, 10))

scatter = plt.scatter(x = 'category', y = 'Achievement', data = df)

#scatter.set(ylim = (0, 200))

plt.show()
plt.figure(figsize = (15, 10))

scatter = plt.scatter(x = 'main_category', y = 'Achievement', data = df)

#scatter.set(ylim = (0, 200))

plt.show()
index = [] 

for i in range(len(df['Achievement'])):

    if df['Achievement'][i] > 0.02:

        index.append(i)

df = df.drop(index, axis=0)

df = df.reset_index()
plt.figure(figsize = (15, 10))

scatter = plt.scatter(x = 'main_category', y = 'Achievement', data = df)

#scatter.set(ylim = (0, 200))

plt.show()
#standizaton_process

#period

import datetime

datetime = copy.deepcopy(df)

datetime_launched = pd.to_datetime(datetime['launched'], format='%Y-%m-%d %H:%M:%S')

datetime_deadline = pd.to_datetime(datetime['deadline'], format='%Y-%m-%d %H:%M:%S')

period = datetime_deadline - datetime_launched

period_df = pd.DataFrame(period.values / np.timedelta64(1, 's'), columns=['period'])



#usd_goal_real

goal = copy.deepcopy(df)

goal_df = pd.DataFrame(goal['usd_goal_real'], columns=['usd_goal_real'])



#standizaton

data_values = pd.concat([period_df, goal_df], axis=1)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

data_values_std = pd.DataFrame(sc.fit_transform(data_values), columns=['period_std','usd_goal_real_std'])



#onehot encording

import copy

onehot = copy.deepcopy(df)

onehot_currency = pd.get_dummies(onehot['currency'])

onehot_country = pd.get_dummies(onehot['country'])

onehot_category = pd.get_dummies(onehot['main_category'])





df_dataset = pd.concat([data_values_std, onehot_country, onehot_category, df['Achievement']], axis=1)# onehot_currencyを一時排除

df_dataset.tail(50)
print(df_dataset.columns)
df_dataset.corr().style.background_gradient().format('{:.2f}')
#stepwise method

from sklearn.feature_selection import RFECV

from sklearn.linear_model import LinearRegression

estimator = LinearRegression(normalize=False)

rfecv = RFECV(estimator, cv=55, scoring='neg_mean_squared_error')

Y = df_dataset['Achievement']

X = df_dataset.drop('Achievement', axis=1)



y = Y.values

X = X.values



# fitで特徴選択を実行

rfecv.fit(X, y)

# 特徴のランキングを表示（1が最も重要な特徴）

print('Feature ranking: \n{}'.format(rfecv.ranking_))
X_data = df_dataset.drop(['Achievement'], axis=1)

Y_data2 = df_dataset['Achievement']
from sklearn.model_selection import train_test_split

#Linear Regression

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X_data, Y_data2, test_size=0.2, random_state=0)

from sklearn.linear_model import LinearRegression

Linreg = LinearRegression()

Linreg = Linreg.fit(X2_train, Y2_train)





#Ridge Regression

from sklearn.linear_model import Ridge

Ridge = Ridge(alpha=0.01)

Ridge.fit(X2_train, Y2_train)
from sklearn.metrics import mean_squared_error, mean_absolute_error

y2_pre1 = Linreg.predict(X2_test)

MSE1 = mean_squared_error(Y2_test, y2_pre1)

print('MSE_lin = ', MSE1)

RMSE1 = np.sqrt(MSE1)

print('RMSE_lin = ', RMSE1)

MAE1 = mean_absolute_error(Y2_test, y2_pre1)

print('MAE_lin = ', MAE1)



y2_pre2 = Ridge.predict(X2_test)

MSE2 = mean_squared_error(Y2_test, y2_pre2)

print('MSE_Ridge = ', MSE2)

RMSE2 = np.sqrt(MSE2)

print('RMSE_Ridge = ', RMSE2)

MAE2 = mean_absolute_error(Y2_test, y2_pre2)

print('MAE_Ridge = ', MAE2)