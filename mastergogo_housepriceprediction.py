# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import os
import re
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
train = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/train.csv')
test = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/test.csv')
sample = pd.read_csv(r'/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')
pd.set_option('display.max_rows', 100)
train.isnull().sum()
train.info()
columns_with_most_nulls = []
for col in train.columns:
    if train[col].isnull().sum()/train.shape[0] > 0.20:
        columns_with_most_nulls.append(col)
print(columns_with_most_nulls)
cal_columns = []
for col in train.columns:
    if train[col].nunique() < 10 and col not in columns_with_most_nulls:
        cal_columns.append(col)
print(cal_columns)
print("{} columns out of {} appear categorical".format(len(cal_columns),train.shape[1]))
for cat_cols in cal_columns:
    print("Column {} has value counts {}".format(cat_cols,train[cat_cols].value_counts()))
train = train.drop(columns_with_most_nulls,axis=1)
test = test.drop(columns_with_most_nulls,axis=1)
for col in tqdm_notebook(train.columns): 
    if train[col].dtype == 'object':
        le = LabelEncoder()
        le.fit(list(train[col].astype(str).values) + list(test[col].astype(str).values))
        train[col] = le.transform(list(train[col].astype(str).values))
        test[col] = le.transform(list(test[col].astype(str).values))
print(train.columns)
print(test.columns)
y=train['SalePrice']
X_train = train.drop('SalePrice',axis=1)
X_test = test.copy()
X_train = X_train.fillna(X_train.mean())
X_test = X_test.fillna(X_test.mean())
from sklearn.model_selection import train_test_split
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y,test_size=.20,random_state=42)
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_squared_error
br = BaggingRegressor(n_estimators=16,random_state=42)
br.fit(X_tr,y_tr)
pred = br.predict(X_val)
print(np.sqrt(mean_squared_error(np.log(y_val),np.log(pred))))
br = BaggingRegressor(n_estimators=16,random_state=42)
br.fit(X_train,y)
pred = br.predict(X_train)
print(np.sqrt(mean_squared_error(np.log(y),np.log(pred))))
print(sample.shape)
print(test.shape)
#X_train.info()

#X_test.info()
X_train.head()
y_pred = br.predict(X_test)
sample['SalePrice'] = y_pred
sample.to_csv('submission.csv',index=False)
y=train['SalePrice']
X_train = train.drop('SalePrice',axis=1)
X_test = test.copy()
for col in X_train.columns:
    if col in cat_cols:
        X_train[col] = X_train[col].fillna(X_train[col].mode())
        X_test[col] = X_test[col].fillna(X_test[col].mode())
    else:
        X_train[col] = X_train[col].fillna(X_train[col].mean())
        X_test[col] = X_test[col].fillna(X_test[col].mean())

X_train.isnull().sum()
X_tr,X_val,y_tr,y_val = train_test_split(X_train,y,test_size=.20,random_state=42)
br = BaggingRegressor(n_estimators=16,random_state=42)
br.fit(X_tr,y_tr)
pred = br.predict(X_val)
print(np.sqrt(mean_squared_error(np.log(y_val),np.log(pred))))
#No difference in changing the impputation method
#lets have a look a columns with nans
X_train.isnull().sum().sort_values(ascending=False)
contains_nan = ['LotFrontage','GarageYrBlt','MasVnrArea']
#only 3 columns contain NaN. Lets visualize them separately
nan_df = X_train[contains_nan]
nan_df.shape
nan_df.describe()
#looks like MasVnrArea has many 0s. lets check
print(np.count_nonzero(nan_df['MasVnrArea'] == 0)/nan_df.shape[0])
#almost 60% of the houses hass this value==0,hence explains its large std dev
#We will create a column to indicate thathouse has MasVnrArea and remove the actual column
X_train['hasMasVnrArea'] = (X_train['MasVnrArea'] > 0 ).astype(int)
X_test['hasMasVnrArea'] = (X_test['MasVnrArea'] > 0 ).astype(int)
del X_train['MasVnrArea']
del X_test['MasVnrArea']
#increased the loss by 0.004. we will try 2nd strateg
