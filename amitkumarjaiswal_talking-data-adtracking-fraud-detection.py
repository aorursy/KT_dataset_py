import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold, RepeatedKFold
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, accuracy_score
import gc

def extractDateFeatures(df, sourceName):
    seasons = [0,0,1,1,1,2,2,2,3,3,3,0] #dec - feb is winter, then spring, summer, fall etc
    df['df_day_' + sourceName] = pd.to_datetime(df[sourceName]).dt.day.astype('uint8')
    df['df_weekday_' + sourceName] = pd.to_datetime(df[sourceName]).dt.dayofweek.astype('uint8')
    df['df_hour_' + sourceName] = pd.to_datetime(df[sourceName]).dt.hour.astype('uint8')

dtypes = {
        'ip'            : 'uint32',
        'app'           : 'uint16',
        'device'        : 'uint16',
        'os'            : 'uint16',
        'channel'       : 'uint16',
        'is_attributed' : 'uint8',
        'click_id'      : 'uint32'
        }    

train=pd.read_csv("../input/train_sample.csv",\
                 dtype=dtypes)
extractDateFeatures(train,'click_time')

GROUPBY_AGGREGATIONS = [
    # Variance in day, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'df_day_click_time', 'agg': 'var', 'type': 'float32'},
    # Variance in day, for ip-app-device
    {'groupby': ['ip','app','device'], 'select': 'df_day_click_time', 'agg': 'var', 'type': 'float32'},
    # Variance in day, for ip-app-os
    {'groupby': ['ip','app','os'], 'select': 'df_day_click_time', 'agg': 'var', 'type': 'float32'},

    # Count, for ip-day-hour
    {'groupby': ['ip','df_day_click_time','df_hour_click_time'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},
    
    # Count, for ip-app
    {'groupby': ['ip', 'app'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},        
    # Count, for ip-app-os
    {'groupby': ['ip', 'app', 'os'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},
    # Count, for ip-app-day-hour
    {'groupby': ['ip','app','df_day_click_time','df_hour_click_time'], 'select': 'channel', 'agg': 'count', 'type': 'uint32'},
    
    # Mean hour, for ip-app-channel
    {'groupby': ['ip','app','channel'], 'select': 'df_hour_click_time', 'agg': 'mean', 'type': 'float32', 'type': 'float32'}
]
test=pd.read_csv("../input/test.csv"\
                 ,nrows=100000, dtype=dtypes)
extractDateFeatures(test,'click_time')
for spec in GROUPBY_AGGREGATIONS:
    # Unique list of features to select
    all_features = list(set(spec['groupby'] + [spec['select']]))
    # Name of new feature
    new_feature = '{}_{}_{}'.format('_'.join(spec['groupby']), spec['agg'], spec['select'])
     # Perform the groupby
    gp = train[all_features]. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature}).astype(spec['type'])
     # Merge back to X_train
    train = train.merge(gp, on=spec['groupby'], how='left')
    
    gp = test[all_features]. \
        groupby(spec['groupby'])[spec['select']]. \
        agg(spec['agg']). \
        reset_index(). \
        rename(index=str, columns={spec['select']: new_feature}).astype(spec['type'])
     # Merge back to X_train
    test = test.merge(gp, on=spec['groupby'], how='left')
    
del gp
gc.collect()

train.fillna(0,inplace=True)
test.fillna(0,inplace=True)
from sklearn.model_selection import train_test_split

y_train = train['is_attributed']
x_train = train.drop(['is_attributed','click_time','attributed_time'],axis=1)
y_test = train['is_attributed']
x_test = test.drop(['click_time'],axis=1)

cnt = 0
p_buf = []
n_splits = 2
n_repeats = 1
kf = RepeatedKFold(
    n_splits=n_splits, 
    n_repeats=n_repeats, 
    random_state=0)
auc_buf = []   

params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'auc',
        'max_depth': 12,
        'num_leaves': 31,
        'learning_rate': 0.025,
        'feature_fraction': 0.85,
        'bagging_fraction': 0.85,
        'bagging_freq': 5,
        'verbose': 0,
        'num_threads': 4,
        'lambda_l2': 1.5,
        'min_gain_to_split': 0,
    }  

for train_index, valid_index in kf.split(x_train):
    print('Fold {}/{}'.format(cnt + 1, n_splits))
    
    model = lgb.train(
        params,
        lgb.Dataset(x_train.loc[train_index], y_train.loc[train_index], feature_name=x_train.columns.tolist()),
        num_boost_round=10000,
        valid_sets=[lgb.Dataset(x_train.loc[valid_index], y_train.loc[valid_index])],
        early_stopping_rounds=100,
        verbose_eval=100,
    )

    if cnt == 0:
        importance = model.feature_importance()
        model_fnames = model.feature_name()
        tuples = sorted(zip(model_fnames, importance), key=lambda x: x[1])[::-1]
        tuples = [x for x in tuples if x[1] > 0]
        print('Important features:')
        print(tuples[:200])

    p = model.predict(x_train.loc[valid_index], num_iteration=model.best_iteration)
    auc = roc_auc_score(y_train.loc[valid_index], p)

    print('{} AUC: {}'.format(cnt, auc))

    p = model.predict(x_test, num_iteration=model.best_iteration)
    if len(p_buf) == 0:
        p_buf = np.array(p)
    else:
        p_buf += np.array(p)
    auc_buf.append(auc)

    cnt += 1
    if cnt > 0: # Comment this to run several folds
        break
    
    '''del model
    gc.collect'''

auc_mean = np.mean(auc_buf)
auc_std = np.std(auc_buf)
print('AUC = {:.6f} +/- {:.6f}'.format(auc_mean, auc_std))

preds = p_buf/cnt
import csv

subm = pd.DataFrame()
subm['click_id'] = test['click_id']
subm['is_attributed'] = preds
subm.to_csv('talkingdata_submission.csv', index=False,quoting=csv.QUOTE_NONNUMERIC)
subm.head()