# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

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

train_df = pd.read_csv("../input/train.csv", skiprows=range(1,123903891), nrows=6100000, usecols=train_columns, dtype=dtypes)
test_df = pd.read_csv("../input/test.csv", usecols=test_columns, dtype=dtypes)

train_y = train_df['is_attributed']
train_df.drop(['is_attributed'], axis=1, inplace=True)
#train_df.drop(['attributed_time'], axis=1, inplace=True)


test_df.head()
# Convert datetime feature to : dayofweek + dayofyear + hour of day

def timeconvert(df):
    df['datetime'] = pd.to_datetime(df['click_time'])
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["dayofyear"] = df["datetime"].dt.dayofyear
    df["hourofday"] = df["datetime"].dt.hour
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    
    return df

# get the nb of click per IP per day

def nb_click_per_ip_per_day(df):
    df['dateclick'] = pd.to_datetime(df['click_time']).dt.date
    return df.groupby(['ip', 'dateclick']).size().reset_index().rename(columns={0:'nbclick'})
    
nrow_train = train_df.shape[0]
merge = pd.concat([train_df, test_df])

# Count the average number of clicks per day by ip
gc.collect()


merge['dateclick'] = pd.to_datetime(merge['click_time']).dt.date
ip_count1 = merge.groupby(['ip', 'dateclick']).size().reset_index().rename(columns={0:'nbclick'})
gc.collect()
ip_count = ip_count1.groupby(['ip']).mean().reset_index()
ip_count.columns = ['ip', 'avg_clicks_per_day_by_ip']
merge = pd.merge(merge, ip_count, on='ip', how='left', sort=False)
#merge.drop('ip', axis=1, inplace=True)
merge.drop('dateclick', axis=1, inplace=True)

train_df = merge[:nrow_train]
test_df = merge[nrow_train:]

merge.head()
train_df.drop('click_id', axis=1, inplace=True)
train_df.head()

#train_df2 = nb_click_per_ip_per_day(train_df)
#train_df2.head()
#train_df2['nbclick'].unique()
#train_df2.loc[train_df2['nbclick'] < 10]
#train_all.loc[train_df['ip'] == 364084].groupby(['is_attributed']).size().reset_index().rename(columns={0:'count'})
#train_y[31]
#Prepare the data

train_df = timeconvert(train_df)
train_df.head()
dtrain = xgb.DMatrix(train_df, train_y)

# Set the params(this params from Pranav kernel) for xgboost model
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

#model_xgb = xgb.XGBClassifier(**params)

watchlist = [(dtrain, 'train')]

model_xgb = xgb.train(params, dtrain, 200, watchlist, maximize=True, early_stopping_rounds = 25, verbose_eval=5)

plot_importance(model_xgb)

'''
results = cross_val_score(model_xgb, train_df, train_y, cv=10)
print("XGB score: %.4f (%.4f)" % (results.mean()*100, results.std()*100))
print(results)
'''
sub_df = pd.DataFrame()
sub_df['click_id'] = test_df['click_id'].astype('int')
test_df.drop(['click_id'], axis=1, inplace=True)
test_df = timeconvert(test_df)
dtest = xgb.DMatrix(test_df)
test_df.head()
# Save the predictions
sub_df['is_attributed'] = model_xgb.predict(dtest)
sub_df.to_csv('xgb_sub.csv', float_format='%.8f', index=False)