import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import gc
import os
cwd = os.getcwd()
train_path = '../input/'+'train.csv'
test_path = '../input/'+'test.csv'
train_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
test_cols = ['ip', 'app', 'device', 'os', 'channel', 'click_time']
dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }
print('Loading the training data...')
train = pd.read_csv(train_path, skiprows=range(1,84903891), dtype=dtypes, nrows=11000000,usecols=train_cols)
train.info()
len_train = len(train)
print('The initial size of the train set is', len_train)
def df_featured(df):
    print("Creating new time features: 'hour' and 'day'...")
    df['hour'] = pd.to_datetime(df.click_time).dt.hour.astype('uint8')
    df['day'] = pd.to_datetime(df.click_time).dt.day.astype('uint8')
    gc.collect()
    
    print("Feature Engineering \n")
    
    print('1. Computing the number of clicks associated with a given IP address within each hour... ')
    n_channel = df[['ip','day','hour','channel']].groupby(by=['ip','day',
          'hour'])[['channel']].count().reset_index().rename(columns={'channel': 'n_channels'})
    print('Merging the channels data with the main data set...\n')
    df = df.merge(n_channel, on=['ip','day','hour'], how='left')
    del n_channel
    gc.collect()
          
          
    print('2. Computing the number of clicks associated with a given IP address and app...')
    n_channel = df[['ip','app', 'channel']].groupby(by=['ip', 
          'app'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_count'})
    print('Merging the channels data with the main data set...\n')
    df = df.merge(n_channel, on=['ip','app'], how='left')
    del n_channel
    gc.collect()
          
    print('3. Computing the number of clicks associated with a given IP address, app, and os...')
    n_channel = df[['ip','app', 'os', 'channel']].groupby(by=['ip', 'app', 
          'os'])[['channel']].count().reset_index().rename(columns={'channel': 'ip_app_os_count'}) 
    print('Merging the channels data with the main data set...\n')       
    df = df.merge(n_channel, on=['ip','app', 'os'], how='left')
    del n_channel
    gc.collect()
          
    # Adding features with var and mean hour (inspired from nuhsikander's script)  
          
    print('4. grouping by : ip_day_chl_var_hour..')
    n_channel = df[['ip','day','hour','channel']].groupby(by=['ip','day',
            'channel'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_tchan_count'})
    print('Merging the hour data with the main data set...\n') 
    df = df.merge(n_channel, on=['ip','day','channel'], how='left')
    del n_channel
    gc.collect()
          
    print('5. grouping by : ip_app_os_var_hour..')
    n_channel = df[['ip','app', 'os', 'hour']].groupby(by=['ip', 'app', 
                'os'])[['hour']].var().reset_index().rename(index=str, columns={'hour': 'ip_app_os_var'})
    print('Merging the hour data with the main data set...\n') 
    df = df.merge(n_channel, on=['ip','app', 'os'], how='left')
    del n_channel
    gc.collect()
          
    print('6. grouping by : ip_app_channel_var_day...')
    n_channel = df[['ip','app', 'channel', 'day']].groupby(by=['ip', 'app', 
                'channel'])[['day']].var().reset_index().rename(index=str, columns={'day': 'ip_app_channel_var_day'})
    print('Merging the day data with the main data set...\n') 
    df = df.merge(n_channel, on=['ip','app', 'channel'], how='left')
    del n_channel
    gc.collect()
          
    print('7. grouping by : ip_app_chl_mean_hour..')
    n_channel = df[['ip','app', 'channel','hour']].groupby(by=['ip', 'app', 
                'channel'])[['hour']].mean().reset_index().rename(index=str, columns={'hour': 'ip_app_channel_mean_hour'})
    print('Merging the meanhour data with the main data set...\n') 
    df = df.merge(n_channel, on=['ip','app', 'channel'], how='left')
    del n_channel
    gc.collect()
    
    print('8. group by : ip_hh_dev')
    n_channel = df[['ip', 'device', 'hour', 'day', 'channel']].groupby(by=['ip', 'device', 'day',
             'hour'])[['channel']].count().reset_index().rename(index=str, columns={'channel': 'nip_hh_dev'})
    print('Merging the channel data with the main data set...\n') 
    df = df.merge(n_channel, on=['ip','device','day','hour'], how='left')
    del n_channel
    gc.collect()     
    df['n_channels'] = df['n_channels'].astype('uint16')
    df['ip_app_count'] = df['ip_app_count'].astype('uint16')
    df['ip_app_os_count'] = df['ip_app_os_count'].astype('uint16')
    df['ip_tchan_count'] = df['ip_tchan_count'].astype('float32')
    df['ip_app_os_var'] = df['ip_app_os_var'].astype('float32')
    df['ip_app_channel_var_day'] = df['ip_app_channel_var_day'].astype('float32')
    df['ip_app_channel_mean_hour'] = df['ip_app_channel_mean_hour'].astype('float32')  
    df['nip_hh_dev'] = df['nip_hh_dev'].astype('uint16') 
    df.drop( ['ip','day'], axis=1, inplace=True )
    gc.collect()
    return df     
print( "Train info before: \n\n")
print( train.info() )
train = df_featured( train )
gc.collect()
print( "Train info after: \n\n")
print( train.info() )
metrics = 'auc'
lgb_params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric':metrics,
        'learning_rate': 0.05,
        'num_leaves': 31,  # we should let it be smaller than 2^(max_depth)
        'max_depth': -1,  # -1 means no limit
        'min_child_samples': 20,  # Minimum number of data need in a child(min_data_in_leaf)
        'max_bin': 255,  # Number of bucketed bin for feature values
        'subsample': 0.7,  # Subsample ratio of the training instance.
        'subsample_freq': 1,  # frequence of subsample, <=0 means no enable
        'colsample_bytree': 0.7,  # Subsample ratio of columns when constructing each tree.
        'subsample_for_bin': 200000,  # Number of samples for constructing bin
        'min_child_weight': 5,  # Minimum sum of instance weight(hessian) needed in a child(leaf)
        'min_split_gain': 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
        'nthread': 8,
        'verbose': 0,
        'scale_pos_weight':99.7, # because training data is extremely unbalanced 
        'metric':metrics
}
target = 'is_attributed'
train[target] = train[target].astype('uint8')

predictors = ['app','device',  'os', 'channel', 'hour', 'n_channels', 'ip_app_count', 'ip_app_os_count',
             'ip_tchan_count','ip_app_os_var','ip_app_channel_var_day','ip_app_channel_mean_hour','nip_hh_dev']
categorical = [ 'app', 'device', 'os', 'channel']
gc.collect()
# print(train.head(5))
VALIDATE = False
MAX_ROUNDS = 1000
EARLY_STOP = 50
OPT_ROUNDS = 1000

FULL_OUTFILE = 'sub_lgbm1.csv'
VALID_OUTFILE = 'sub_lgbm_v.csv'


if VALIDATE:
    train, val_df = train_test_split( train, train_size=.95, shuffle=False )

    print("\nTrain Information ", train.info())
    print("\nVal Information ", val_df.info())

    print("train size: ", len(train))
    print("valid size: ", len(val_df))
    gc.collect()

    print("Training...\n")

    num_boost_round=MAX_ROUNDS
    early_stopping_rounds=EARLY_STOP

    xgtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train
    gc.collect()

    xgvalid = lgb.Dataset(val_df[predictors].values, label=val_df[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del val_df
    gc.collect()

    evals_results = {}

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     valid_sets= [xgvalid], 
                     valid_names=['valid'], 
                     evals_result=evals_results, 
                     num_boost_round=num_boost_round,
                     early_stopping_rounds=early_stopping_rounds,
                     verbose_eval=10, 
                     feval=None)

    n_estimators = bst.best_iteration

    print("\nModel Report")
    print("n_estimators : ", n_estimators)
    print(metrics+":", evals_results['valid'][metrics][n_estimators-1])
    
    outfile = VALID_OUTFILE
    
    del xgvalid

else:

    

    print("train size: ", len(train))

    gc.collect()

    print("Training...")

    num_boost_round=OPT_ROUNDS

    xgtrain = lgb.Dataset(train[predictors].values, label=train[target].values,
                          feature_name=predictors,
                          categorical_feature=categorical
                          )
    del train
    gc.collect()
    print ("Dataset preparing done")

    bst = lgb.train(lgb_params, 
                     xgtrain, 
                     num_boost_round=num_boost_round,
                     verbose_eval=10, 
                     feval=None)
    print("Traing done")
    outfile = FULL_OUTFILE

del xgtrain
gc.collect()

print('load test...')
test_cols = ['ip','app','device','os', 'channel', 'click_time', 'click_id']
test_df = pd.read_csv(test_path, dtype=dtypes, usecols=test_cols)

test_df = df_featured( test_df )
gc.collect()

sub = pd.DataFrame()
sub['click_id'] = test_df['click_id']

print("Predicting...")
sub['is_attributed'] = bst.predict(test_df[predictors])
print("writing...")
sub.to_csv(outfile, index=False, float_format='%.9f')
print("done...")
print(sub.info())