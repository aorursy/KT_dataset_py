# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
train_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test_df = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
test_df['Id']
y = train_df['SalePrice']
y = pd.DataFrame(y)
y
train_df = train_df.drop('SalePrice', axis=1)
all_df = [train_df, test_df]
all_dfs = pd.concat(all_df).reset_index(drop=True)
all_dfs
dropped_df = all_dfs.dropna(axis=1)
dropped_df.shape
all_dfs = all_dfs.dropna(axis=1)
all_dfs = all_dfs.drop('Id', axis=1)
all_dfs
Scaler = StandardScaler()
all_dfs = pd.get_dummies(all_dfs)
all_scaled = pd.DataFrame(Scaler.fit_transform(all_dfs))

train_scaled = all_scaled[:1460]
test_scaled = all_scaled[1460:]
test_scaled
print(train_scaled.shape)
print(test_scaled.shape)
XGB = XGBRegressor(max_depth=3,learning_rate=0.1,n_estimators=1000,reg_alpha=0.001,reg_lambda=0.000001,n_jobs=-1,min_child_weight=3)
LGBM = LGBMRegressor(n_estimators=1000)
X_train, X_test, y_train, y_test = train_test_split(train_scaled, y, test_size=.1, random_state=42)

XGB.fit(X_train, y_train)
LGBM.fit(X_train, y_train)
print('Training score: ',XGB.score(X_train, y_train), 'Test score: ', XGB.score(X_test, y_test))
print('Training score: ',LGBM.score(X_train, y_train), 'Test score: ', LGBM.score(X_test, y_test))
y_pred_xgb = pd.DataFrame(XGB.predict(test_scaled))
y_pred_lgbm = pd.DataFrame(LGBM.predict(test_scaled))

y_pred = pd.DataFrame()
y_pred['Id'] = test_df['Id']
y_pred['SalePrice'] = y_pred_lgbm[0]*0.45 + y_pred_xgb[0]*0.55
y_pred = y_pred.set_index('Id')
y_pred
y_pred.to_csv('sub_blend.csv')