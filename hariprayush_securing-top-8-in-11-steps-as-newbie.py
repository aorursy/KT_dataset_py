# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.



train_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/train.csv')

test_data = pd.read_csv('/kaggle/input/home-data-for-ml-course/test.csv')
null_columns = [c for c in train_data.columns if train_data[c].isnull().sum() > (train_data.shape[0]/2)]

print(null_columns)

X_train = train_data.drop(null_columns, axis = 1)

X_test = test_data.drop(null_columns, axis = 1)
X_train = X_train.drop(['Id'], axis = 1)
X_num_col = X_train.select_dtypes(include=['int64','float'])

X_cat_col = X_train.select_dtypes(include=['object'])
X_num_col.hist(figsize=(20, 20), bins=100, xlabelsize=8, ylabelsize=8);
import seaborn as sns

for i in range(0, len(X_train.columns), 5):

    sns.pairplot(data=X_train,

                x_vars=X_train.columns[i:i+5],

                y_vars='SalePrice')
y = X_num_col['SalePrice']

X_num_col.drop(['SalePrice'],axis=1, inplace=True)
X_num_col.fillna(X_num_col.mean(), inplace=True)



X_test[X_num_col.columns].fillna(X_test[X_num_col.columns].mean(), inplace=True)
X_cat_col.fillna('NA', inplace=True)

X_test[X_cat_col.columns].fillna('NA', inplace=True)
import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (20,16))

sns.heatmap(X_num_col.corr(), annot=True)
X_num_col['GrLivArea'] = (X_num_col['GrLivArea'] / X_num_col['TotRmsAbvGrd'])

X_num_col.drop(['TotRmsAbvGrd'], axis = 1, inplace=True)



X_num_col['GarageArea'] = X_num_col['GarageArea'] + X_num_col['GarageCars']

X_num_col.drop(['GarageCars'], axis = 1, inplace=True)



X_test['GrLivArea'] = X_test['GrLivArea'] / X_test['TotRmsAbvGrd']

X_test['GarageArea'] = X_test['GarageArea'] + X_test['GarageCars']

X_test.drop(['TotRmsAbvGrd', 'GarageCars'], axis=1, inplace=True)

Xt = pd.get_dummies(pd.concat([X_num_col, X_cat_col],axis=1, sort=False))

Xtst = pd.get_dummies(X_test)

X_tr, X_test = Xt.align(Xtst, join='left', axis=1)
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(Xt, y, train_size=0.8, test_size=0.2, random_state=0)



from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error



lin_model = LinearRegression()

lin_model.fit(X_tr, y_tr)

lin_pred = lin_model.predict(X_val)

lin_mean = mean_absolute_error(y_val, lin_pred)

lin_mean
from xgboost import XGBRegressor



xbr_model = XGBRegressor(n_estimators=500, learning_rate=0.06, random_state=5) 

xbr_model.fit(X_tr, y_tr)

xbr_preds = xbr_model.predict(X_val)

xbr_mea = mean_absolute_error(y_val, xbr_preds)

xbr_mea
from sklearn.ensemble import GradientBoostingRegressor



gbr_model = GradientBoostingRegressor(n_estimators=1250, learning_rate=0.04) 

gbr_model.fit(X_tr, y_tr)

gbr_pred = gbr_model.predict(X_val)

gbr_mean = mean_absolute_error(y_val, gbr_pred)

gbr_mean
test_preds= xbr_model.predict(X_test)

output = pd.DataFrame({'Id': test_data['Id'],

                       'SalePrice': test_preds})

output.to_csv('submission.csv', index=False)