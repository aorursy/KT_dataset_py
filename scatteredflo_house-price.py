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
pd.set_option('max_columns',130, 'max_rows', 130)
train = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/train.csv')

test = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/test.csv')

sample_submission = pd.read_csv('/kaggle/input/house-prices-advanced-regression-techniques/sample_submission.csv')

train.shape, test.shape
sample_submission
train.head()
train['MSZoning'].unique()

test.head()
train.isnull().sum() / len(train)
train.dtypes
train_dtypes = pd.DataFrame(train.dtypes, columns = ['type'])

train_dtypes2 = train_dtypes[train_dtypes['type'] == 'object']

train_dtypes3 = train_dtypes2.reset_index()

train_dtypes3['index']
train = train.drop(train_dtypes3['index'],1)
test = test.drop(train_dtypes3['index'], 1)

test.head()

# label encoding
y = train['SalePrice']

X = train.drop(['SalePrice'],1)



X.shape, y.shape
from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state = 0, test_size=0.2)



X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
import lightgbm as lgb

model = lgb.LGBMRegressor(n_estimators=3300, bagging_fraction=0.7, 

                          learning_rate=0.1,

                         max_depth=6, subsample=0.7, 

                          feature_fraction=0.9, boosting_type='gbdt',

                         colsample_bytree=0.5, 

                          reg_lambda=5, n_jobs=-1)





model.fit(X_train,y_train)
pred_train = model.predict(X_train)

pred_valid = model.predict(X_valid)
from sklearn.metrics import mean_squared_error

mean_squared_error(y_train, pred_train)**0.5
pred_train
import matplotlib.pyplot as plt ########



y_valid = pd.DataFrame(y_valid)

y_valid['pred'] = pred_valid

y_valid = y_valid.reset_index(drop=True)



plt.figure(figsize = (100,8))    #########   

y_valid['pred'].T.plot()

y_valid['SalePrice'].T.plot()

mean_squared_error(y_valid, pred_valid)**0.5

pred_test = model.predict(test)
pred_test
sample_submission['SalePrice'] = pred_test

sample_submission.to_csv('./final_submission.csv')