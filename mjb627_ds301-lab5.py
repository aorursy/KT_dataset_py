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
df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
df
#df = df.replace(np.NAN, 'None')

df.head()
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor

from math import sqrt

import seaborn as sns

import matplotlib.pyplot as plt
# use lots of features for XGBRegressor

# fill or drop N/A values, sklearn.impute to fill with median/mean

# for one hot encoding, concat train and test, add new column 0 for train, 1 for test. do one hot then split by 0 and 1
df.info()
#Correlation matrix

corrmat = df.corr()

f, ax = plt.subplots(figsize=(12, 9))

sns.heatmap(corrmat, vmax=.8, square=True);
# Correlation matrix for sale price

k = 10 

cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index

cm = np.corrcoef(df[cols].values.T)

sns.set(font_scale=1.25)

hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)

plt.show()
# Choosing some categorical variables that I think might be important

df['SaleCondition'].value_counts()
df['KitchenQual'].value_counts()
df['HouseStyle'].value_counts()
df['Neighborhood'].value_counts()
df['Foundation'].value_counts()
# One hot encode selected features

df1 = pd.concat([df, pd.get_dummies(df['KitchenQual'], prefix='KitchenQual'), pd.get_dummies(df['HouseStyle'], prefix='HouseStyle'), pd.get_dummies(df['Neighborhood'], prefix='Neighborhood'), pd.get_dummies(df['Foundation'], prefix='Foundation')], axis=1)

drop_cols = ['KitchenQual','HouseStyle','Neighborhood','Foundation']

df1 = df1.drop(drop_cols, axis=1)

test1 = pd.concat([test, pd.get_dummies(test['KitchenQual'], prefix='KitchenQual'), pd.get_dummies(test['HouseStyle'], prefix='HouseStyle'), pd.get_dummies(test['Neighborhood'], prefix='Neighborhood'), pd.get_dummies(test['Foundation'], prefix='Foundation')], axis=1)

test1 = test1.drop(drop_cols, axis=1)
df1.head()
# Verify one hot encoder worked

print(df.shape)

print(df1.shape)

print(test.shape)

print(test1.shape)
# Modeling

X = df1.iloc[:,[15,35,40,43,57,58,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113]]

y = df1['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# RandomForestRegressor

rnd_reg = RandomForestRegressor(n_estimators=500, max_leaf_nodes=16, n_jobs=1)

rnd_reg.fit(X_train, y_train)

y_pred_rf = rnd_reg.predict(X_test)
print("RandomForestRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_rf)))
# Ada boost

from sklearn.ensemble import AdaBoostRegressor

from sklearn.tree import DecisionTreeRegressor
ada_reg = AdaBoostRegressor(DecisionTreeRegressor(max_depth=5), n_estimators=200, learning_rate=0.5)

ada_reg.fit(X_train, y_train)

y_pred_ada=ada_reg.predict(X_test)
print("AdaBoostRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_ada)))
# XG boost
from xgboost import XGBRegressor
xgb_reg = XGBRegressor()

xgb_reg.fit(X_train, y_train)

y_pred_xgb = xgb_reg.predict(X_test)
print("XGBRegressor RMSE:",sqrt(mean_squared_error(y_test, y_pred_xgb)))
for cols in df1.columns:

    print(cols,df1.columns.get_loc(cols))

for x in range(77, 114):

    print(x, end=',')