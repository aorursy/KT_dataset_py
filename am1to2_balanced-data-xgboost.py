import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
import xgboost as xgb

import gc
import matplotlib.pyplot as plt
path = '../input/' 
path_train = path + 'train.csv'
path_test = path + 'test.csv'

#nsamples = 100000
unbalance_fact = 10

train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time' ]

dtypes = {
        'ip'            : 'uint64',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint64'
        }
print("Loading Data")
train = pd.read_csv(path_train, usecols=train_cols, dtype=dtypes, parse_dates=["click_time"])
print("Loading is done")
print(len(train))
train_pos = train[train["is_attributed"] == 1]
num_attr = len(train_pos)
print("Number of attributed: {:d}".format(num_attr))
train_neg = train[train["is_attributed"] == 0].sample(num_attr )
train = pd.concat([train_pos, train_neg])
gc.collect()
def prepare_dataframe(df):
    print("Display frames:")
    display(df.head())
    display(df.dtypes)
    display(df.shape)
    
    print("Creating new time features:")
    df['hour'] = df["click_time"].dt.hour.astype('uint8')
    df['day'] = df["click_time"].dt.day.astype('uint8')
    df["minute"] = df["click_time"].dt.minute.astype('uint8')
    
    print("## Get counts per cat.")
    n_chans = df[['ip','day','hour','channel']].groupby(by=['ip','day',
          'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_day_hour'})
    df = df.merge(n_chans, on=['ip','day','hour'], how='left')
    del n_chans
    gc.collect()
    
    n_chans = df[['ip','app', 'channel']].groupby(by=['ip', 
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
    df = df.merge(n_chans, on=['ip','app'], how='left')
    del n_chans
    gc.collect()
    
    n_chans = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 
          'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'})
    df = df.merge(n_chans, on=['ip','app', 'os'], how='left')
    del n_chans
    gc.collect()
    
    n_chans = df[['ip','channel']].groupby(by=['ip'])[['channel']].count().reset_index().rename(columns={'channel': 'count_by_ip'})
    print('Merging the channels data with the main data set...')
    df = df.merge(n_chans, on=['ip'], how='left')

    # Count by IP HOUR CHANNEL
    n_chans = df[['ip','hour','channel','os']].groupby(by=['ip','hour','channel'
               ])[['os']].count().reset_index().rename(columns={'os': 'ip_hour_channel'})
    df = df.merge(n_chans, on=['ip','hour','channel'], how='left')
    del n_chans
    gc.collect()

    # Count by IP HOUR Device
    n_chans = df[['ip','hour','channel','os']].groupby(by=['ip','hour','os'
               ])[['channel']].count().reset_index().rename(columns={'channel': 'ip_hour_os'})
    df = df.merge(n_chans, on=['ip','hour','os'], how='left')
    del n_chans
    gc.collect()

    n_chans = df[['ip','hour','channel','app']].groupby(by=['ip','hour','app'
               ])[['channel']].count().reset_index().rename(columns={'channel': 'ip_hour_app'})
    df = df.merge(n_chans, on=['ip','hour','app'], how='left')
    del n_chans
    gc.collect()

    n_chans = df[['ip','hour','channel','device']].groupby(by=['ip','hour','device'
               ])[['channel']].count().reset_index().rename(columns={'channel': 'ip_hour_device'})
    df = df.merge(n_chans, on=['ip','hour','device'], how='left')
    del n_chans
    gc.collect()
    
    print("Adjusting the data types of the new count features... ")
    df.info()
    df['ip_day_hour'] = df['ip_day_hour'].astype('uint8')
    df['ip_app_count'] = df['ip_app_count'].astype('uint8')
    df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint8')

    # Added..
    df['count_by_ip'] = df['count_by_ip'].astype('uint16')
    df['ip_hour_channel'] = df['ip_hour_channel'].astype('uint16')
    df['ip_hour_os'] = df['ip_hour_os'].astype('uint16')
    df['ip_hour_app'] = df['ip_hour_app'].astype('uint16')
    df['ip_hour_device'] = df['ip_hour_device'].astype('uint16')
    
    return df
    
train = prepare_dataframe(train)
train_df, test_df = train_test_split(train, test_size=0.2, stratify=train["is_attributed"])
train_df, valid_df = train_test_split(train_df, test_size=0.2, stratify=train_df["is_attributed"])
print(len(train_df))
print(len(valid_df))
print(len(test_df))
del train
gc.collect()
print(len(train_df))
print(len(valid_df))
print(len(test_df))

print(train_df["is_attributed"].sum())
print(valid_df["is_attributed"].sum())
print(test_df["is_attributed"].sum())
target = 'is_attributed'

predictors = ['ip', 'device', 'app', 'os', 'channel', 'hour', "minute", # Starter Vars, Then new features below
              'ip_day_hour','count_by_ip','ip_app_count', 'ip_app_os_count',
              "ip_hour_channel", "ip_hour_os", "ip_hour_app","ip_hour_device"]
gc.collect()
# params = {'eta': 0.3,
#           'tree_method': "hist",
#           'grow_policy': "lossguide",
#           'max_leaves': 1400,  
#           'max_depth': 0, 
#           'subsample': 0.9, 
#           'colsample_bytree': 0.7, 
#           'colsample_bylevel':0.7,
#           'min_child_weight':0,
#           'alpha':4,
#           'objective': 'binary:logistic', 
#           'scale_pos_weight':9,
#           'eval_metric': 'auc', 
#           'nthread':8,
#           'random_state': 99, 
#           'silent': True}

params = {'eta': 0.1,
          'objective': 'binary:logistic', 
          'scale_pos_weight':1.0 / unbalance_fact,
          'eval_metric': 'auc', 
          'nthread':8,
          'silent': True}

print(train_df[target].nunique())
print(pd.unique(train_df[target]))
dtrain = xgb.DMatrix(train_df[predictors], train_df[target])
dvalid = xgb.DMatrix(valid_df[predictors], valid_df[target])
#del train_df, valid_df 
gc.collect()
watchlist = [(dtrain, 'train'), (dvalid, 'valid')]
model = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds = 50, verbose_eval=5)
# Nick's Feature Importance Plot
f, ax = plt.subplots(figsize=[7,10])
xgb.plot_importance(model, ax=ax, max_num_features=len(predictors))
plt.title("XGboost Feature Importance")
plt.savefig('feature_import.png')
## Testing accuracy on test split.
print("Testing against test split")
dtest = xgb.DMatrix(test_df[predictors])
test_res = model.predict(dtest, ntree_limit=model.best_ntree_limit)
test_score = roc_auc_score(test_df[target].values, test_res)
print("Test score = {:f}".format(test_score))
print("Loading test data for generating submission.")
test = pd.read_csv(path_test, dtype=dtypes, parse_dates=["click_time"])
test = prepare_dataframe(test)

print("Preparing data for submission...")
dtest = xgb.DMatrix(test[predictors])
test['is_attributed'] = model.predict(dtest, ntree_limit=model.best_ntree_limit)

print("Writing the submission data into a csv file...")
test[["click_id","is_attributed"]].to_csv("submission_xgb_v2.csv",index=False)
print("All Done...")
