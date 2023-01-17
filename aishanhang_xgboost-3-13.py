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
import time
import xgboost as xgb
from sklearn.cross_validation import train_test_split
start_time=time.time()
columns=['ip','app','device','os','channel','click_time','is_attributed']
dtypes={
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
}
train=pd.read_csv('../input/train.csv',skiprows=range(1,149903891),nrows=35000000,usecols=columns,dtype=dtypes)
test=pd.read_csv('../input/test.csv')
print('[{}] Finished to load data'.format(time.time() - start_time))
sub = pd.DataFrame()
sub['click_id'] = test['click_id']
def dataPreProcessTime(df):
    # Transform click_time in two columns(one with date and another with time)
    df['date_click'] = pd.to_datetime(df['click_time']).dt.date
    df['date_click'] = df['date_click'].apply(lambda x: x.strftime('%Y%m%d')).astype(int)
    
    df['time_click'] = pd.to_datetime(df['click_time']).dt.time
    df['time_click'] = df['time_click'].apply(lambda x: x.strftime('%H%M%S')).astype(int)   
    
    df.drop('click_time', axis=1, inplace=True)
    return df

#数据的统计信息
print(train['is_attributed'].value_counts())
print(train[train['is_attributed']==1]['is_attributed'].sum()/len(train))
train = dataPreProcessTime(train)
test = dataPreProcessTime(test)
y=train['is_attributed']
#'click_time','is_attributed','attributed_timed'
train.drop(['is_attributed'],axis=1,inplace=True)#inplace=True代表更改原内存的值
#'click_id','click_time'
test.drop(['click_id'],axis=1,inplace=True)
# Some feature engineering
nrow_train = train.shape[0]
merge = pd.concat([train, test])
# Count the number of clicks by ip
ip_count = merge.groupby('ip')['app'].count().reset_index()
ip_count.columns = ['ip', 'clicks_by_ip']
ip_count.tail()
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
merge.drop('ip', axis=1, inplace=True)
train = merge[:nrow_train]
test = merge[nrow_train:]
test.head()
# Set the params(this params from Pranav kernel) for xgboost model
params = {'eta': 0.6,
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
watchlist = [(xgb.DMatrix(train, y), 'train')]
model = xgb.train(params, xgb.DMatrix(train, y), 15, watchlist, maximize=True, verbose_eval=1)
print('[{}] Finish XGBoost Training'.format(time.time() - start_time))
sub['is_attributed'] = model.predict(xgb.DMatrix(test), ntree_limit=model.best_ntree_limit)
sub.to_csv('xgb_sub.csv',index=False)
sub['is_attributed'].head()