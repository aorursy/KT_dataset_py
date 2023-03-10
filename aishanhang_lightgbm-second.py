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
from sklearn.cross_validation import train_test_split
import lightgbm as lgb
import time
def lgb_modelfit_nocv(params,dtrain,dvalid,predictors,target='target',objective='binary',metrics='auc',feval=None,early_stopping_rounds=20,
                     num_boost_round=3000,verbose_eval=10,categorical_features=None):
    lgb_params={
        'boosting_type':'gbdt',
        'objective':objective,
        'metric':metrics,
        'learning_rate':0.01,
        #'is_unbalance':'true',这里数据不平衡
        'num_leaves':31,#需要让他小于 2^(max_depth)
        'max_depth':-1,
        'min_child_samples':20,#在子集中所需最小数据量
        'max_bin':255,#Number of bucketed bin for feature values
        'subsample':0.6,
        'subsample_freq':0,#子采样频率
        'colsample_bytree':0.3,#列采样比率
        'min_child_weight':5,#孩子需要的实例权重（hessian）的最小总和（叶子）
        'subsample_for_bin':200000,#构建垃圾桶的样本数量
        'min_split_gain':0,
        'reg_alpha':0,#L1 regularization term on weights
        'reg_lambda':0,#L2 regularization term on weights
        'nthread':8,
        'verbose':0
    }
    lgb_params.update(params)
    print('preparing valildation datasets')
    
    xgtrain=lgb.Dataset(dtrain[predictors].values,label=dtrain[target].values,feature_name=predictors,
                        categorical_feature=categorical_features)
    xgvalid=lgb.Dataset(dvalid[predictors].values,label=dvalid[target].values,feature_name=predictors,
                       categorical_feature=categorical_features)
    
    evals_results={}
    
    bst1=lgb.train(lgb_params,xgtrain,valid_sets=[xgtrain,xgvalid],valid_names=['trian','valid'],
                  evals_result=evals_results,num_boost_round=num_boost_round,early_stopping_rounds=early_stopping_rounds,
                  verbose_eval=10,feval=feval)
    n_estimators=bst1.best_iteration
    print('Model Report')
    print('n_estimators:',n_estimators)
    print(metrics+':',evals_results['valid'][metrics][n_estimators-1])
    return bst1
path='../input/'
dtypes={
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
}
print('load train...')#40000000
train_df = pd.read_csv(path+"train.csv",skiprows=range(1,149903891), nrows=40000000, dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'is_attributed'])
print('load test...')
test_df = pd.read_csv(path+"test.csv", dtype=dtypes, usecols=['ip','app','device','os', 'channel', 'click_time', 'click_id'])
import gc
len_train = len(train_df)
train_df=train_df.append(test_df)
del test_df
gc.collect()
print('data prep...')
train_df['hour'] = pd.to_datetime(train_df.click_time).dt.hour.astype('uint8')
train_df['day'] = pd.to_datetime(train_df.click_time).dt.day.astype('uint8')

gc.collect()
train_df.head()
#组合特征 1
print('group by...')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'qty'})
gp.tail()
print('merge...')
train_df = train_df.merge(gp, on=['ip','day','hour'], how='left')

print("vars and data type: ")
train_df.info()
train_df.tail()
#组合特征 2
print('grouping by ip-app combination...')
gp = train_df[['ip', 'app', 'channel']].groupby(by=['ip', 'app'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_count'})
train_df = train_df.merge(gp, on=['ip','app'], how='left')
del gp
gc.collect()
#组合特征 3
print('grouping by ip-app-os combination...')
gp = train_df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 'os'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'ip_app_os_count'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()
# 4
print('grouping by : ip_day_chl_var_hour')
gp = train_df[['ip','day','hour','channel']].groupby(by=['ip','day','channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
train_df = train_df.merge(gp, on=['ip','day','channel'], how='left')
del gp
gc.collect()
# 5
print('grouping by : ip_app_os_var_hour')
gp = train_df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
train_df = train_df.merge(gp, on=['ip','app', 'os'], how='left')
del gp
gc.collect()
# 6
print('grouping by : ip_app_channel_var_day')
gp = train_df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
train_df = train_df.merge(gp, on=['ip','app', 'channel'], how='left')
del gp
gc.collect()
train_df['qty'] = train_df['qty'].astype('uint16')
train_df['ip_app_count'] = train_df['ip_app_count'].astype('uint16')
train_df['ip_app_os_count'] = train_df['ip_app_os_count'].astype('uint16')
test_df = train_df[len_train:]
val_df = train_df[(len_train-3000000):len_train]
train_df = train_df[:(len_train-3000000)]
#train_df = train_df[:len_train]
print("train size: ", len(train_df))
print("valid size: ", len(val_df))
print("test size : ", len(test_df))
target = 'is_attributed'
predictors = ['app','device','os', 'channel', 'hour', 'qty', 
              'ip_tchan_count', 'ip_app_count',
              'ip_app_os_count', 'ip_app_os_var',
              'ip_app_channel_var_day']
categorical = ['app','device','os', 'channel', 'hour']


sub = pd.DataFrame()
sub['click_id'] = test_df['click_id'].astype('int')

gc.collect()
print("Training...")
params = {
    'learning_rate': 0.1,
    #'is_unbalance': 'true', # replaced with scale_pos_weight argument
    'num_leaves': 1400,  # we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  # -1 means no limit
    'min_child_samples': 200,#100  # Minimum number of data need in a child(min_data_in_leaf)
    'max_bin': 100,  # Number of bucketed bin for feature values
    'subsample': .7,  # Subsample ratio of the training instance.
    'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
    'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
    'min_child_weight': 0,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
    'scale_pos_weight':99 # because training data is extremely unbalanced 
}
bst = lgb_modelfit_nocv(params, 
                        train_df, 
                        val_df, 
                        predictors, 
                        target, 
                        objective='binary', 
                        metrics='auc',
                        early_stopping_rounds=50, 
                        verbose_eval=True, 
                        num_boost_round=400, #300
                        categorical_features=categorical)
del train_df
del val_df
gc.collect()

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('sub_lgb_balanced99.csv',index=False)
print("done...")
print(sub.info())
'''print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv('lgb_second.csv',index=False)
print("done...")
print(sub.info())'''