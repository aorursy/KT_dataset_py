# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 

import seaborn as sns

import matplotlib.pyplot as plt



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

train.head(10)
train['SalePrice'].describe()

sns.distplot(train['SalePrice'], color='red')

train.drop(['Id', 'SalePrice'], axis=1).info()

categorical_columns = [col for col in train.columns if train[col].dtype.name == 'object']

numerical_columns   = [col for col in train.columns if train[col].dtype.name != 'object']
nans = []

for col in train.columns:

    if len(train[train[col].isna()]) > 500 or len(train[train[col].isnull()]) > 500:

        nans.append(col)

nans
train = train.drop(nans, axis=1)
for num_col in numerical_columns:

    train[num_col].fillna(train[num_col].mean(), inplace=True)
train.head()

corre = train.corr()

top_corr_features = corre.index[abs(corre['SalePrice'])>0.5]

g = sns.heatmap(train[top_corr_features].corr(),annot=True)
correlationshigh = []

for i in corre:

    for j in corre.index[corre[i] > 0.80]:

        if i != j and j not in correlationshigh and i not in correlationshigh:

            correlationshigh.append(j)

correlationshigh
train = train.drop(correlationshigh, axis=1)
corr_df = train.corr().abs()

corr_df['SalePrice'].sort_values(ascending=False).head(5)
sns.catplot(x='OverallQual', y='SalePrice', data=train)
# GrLivArea/SalePrice

sns.relplot(x='GrLivArea', y='SalePrice', color='red',data=train)
# GrLivArea/SalePrice

sns.relplot(x='TotalBsmtSF', y='SalePrice', color='blue',data=train)
sns.set()

cols = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars', 'TotalBsmtSF', 'FullBath', 'YearBuilt']

sns.pairplot(train[cols], size = 2.5)

plt.show()
train = pd.get_dummies(train.drop(['Id'], axis=1)).reset_index(drop=True)
needed_cols = train.drop(['SalePrice'], axis=1).columns

test = pd.get_dummies(test)

test_columns = test.columns



for needed_col in needed_cols:

    if needed_col not in test_columns:

        test[needed_col] = 0



Ids = test['Id']

test = test[needed_cols]
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train.drop(["SalePrice"], axis=1), train["SalePrice"], test_size = 0.2, random_state = 0)
from sklearn.linear_model import LinearRegression

linear_regressor = LinearRegression()

linear_regressor.fit(X_train, y_train)
y_pred = linear_regressor.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score

print('mean_squared_error: ',mean_squared_error(y_test, y_pred),

     '\nr2_score: ',r2_score(y_test, y_pred))
from xgboost import XGBRegressor



xgb_regressor = XGBRegressor(nthread=-1, learning_rate=0.01, max_depth=5, n_estimators=6000, gamma=0.05, subsample=0.7, objective='reg:linear')

xgb_regressor.fit(train.drop(["SalePrice"], axis=1), train["SalePrice"], verbose=False)
from sklearn.ensemble import RandomForestRegressor

rf_regressor = RandomForestRegressor(max_depth=10, n_estimators=2000)

rf_regressor.fit(train.drop(["SalePrice"], axis=1), train["SalePrice"])
prediction_xgb = xgb_regressor.predict(test)

prediction_rf = rf_regressor.predict(test.fillna(test.mean()))

submit = pd.DataFrame()

submit['Id'] = Ids

submit['SalePrice'] = pd.Series(prediction_xgb)

submit.to_csv('submission_house_prices_prediction_xgb.csv', index=False)
submit = pd.DataFrame()

submit['Id'] = Ids

submit['SalePrice'] = pd.Series(prediction_rf)

submit.to_csv('submission_house_prices_random_forest.csv', index=False)