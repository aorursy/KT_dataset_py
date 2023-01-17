# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
#thanks Joao for your great notebook!

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import time
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
is_valid = False
start_time = time.time()
train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_columns  = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'click_id']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
train=pd.read_csv("../input/train.csv", skiprows=range(1,129903891), nrows=61000000, usecols=train_columns, dtype=dtypes)
test=pd.read_csv("../input/test_supplement.csv",usecols=test_columns, dtype=dtypes)
print('loading of data is completed in [{}] seconds'.format(time.time() - start_time))
def datatimeFeatures(df):
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df['doy']      = df['datetime'].dt.dayofyear
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df
#select the target variable
y = train['is_attributed']

train.drop(['is_attributed'], axis=1, inplace=True)
sub = pd.DataFrame()
test.drop('click_id', axis=1, inplace=True)
gc.collect()

# Some feature engineering
nrow_train = train.shape[0]
merge = pd.concat([train, test])
del train, test
gc.collect()

# Count the number of clicks by ip and app
ip_count = merge.groupby(['ip'])['channel'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
merge['clicks_by_ip'] = merge['clicks_by_ip'].astype('uint16')
merge.drop('ip', axis=1, inplace=True)

train = merge[:nrow_train]
test = merge[nrow_train:]
del test, merge
gc.collect()

# Make new feature with datatimeFeatures function
train = datatimeFeatures(train)
gc.collect()

print('preprocessing is completed in [{}] seconds'.format(time.time() - start_time))
from sklearn.model_selection import train_test_split
import xgboost as xgb
params = {'eta': 0.3,
          'tree_method': "hist",
          'grow_policy': "lossguide",
          'max_leaves': 1400,  
          'max_depth': 0, 
          'subsample': 0.9, 
          'colsample_bytree': 0.7, 
          'colsample_bylevel':0.7,
          'min_child_weight':0,
          'alpha':4,
          'objective': 'binary:logistic', 
          'scale_pos_weight':9,
          'eval_metric': 'auc', 
          'nthread':8,
          'random_state': 99, 
          'silent': True}
if (is_valid == True):
    # Get 10% of train dataset to use as validation
    x1, x2, y1, y2 = train_test_split(train, y, test_size=0.1, random_state=99)
    dtrain = xgb.DMatrix(x1, y1)
    dvalid = xgb.DMatrix(x2, y2)
    del x1, y2, x2, y2 
    gc.collect()
    watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
    model = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)
    del dvalid
else:
    dtrain = xgb.DMatrix(train, y)
    del train, y
    gc.collect()
    watchlist = [(dtrain, 'train')]
    model = xgb.train(params, dtrain, 30, watchlist, maximize=True, verbose_eval=1)

del dtrain
gc.collect()
print('XGBoost Training is finished in [{}] seconds'.format(time.time() - start_time))
# Load the test for predict 
test = pd.read_csv("../input/test.csv", usecols=test_columns, dtype=dtypes)
test = pd.merge(test, ip_count, on='ip', how='left', sort=False)
del ip_count
gc.collect()
sub['click_id'] = test['click_id'].astype('int')
test['clicks_by_ip'] = test['clicks_by_ip'].astype('uint16')
test = datatimeFeatures(test)
test.drop(['click_id', 'ip'], axis=1, inplace=True)
dtest = xgb.DMatrix(test)
del test
gc.collect()
# Save the predictions
sub['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)
sub.to_csv('improved_xgb_sub_today.csv',float_format='%.8f',index=False)
print('submission is done in [{}] seconds'.format(time.time() - start_time))
