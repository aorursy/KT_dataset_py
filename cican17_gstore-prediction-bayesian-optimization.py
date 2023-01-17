import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import time
from datetime import datetime
import gc
import psutil
from sklearn.preprocessing import LabelEncoder

PATH="../input/"
NUM_ROUNDS = 20000
VERBOSE_EVAL = 500
STOP_ROUNDS = 100
N_SPLITS = 10

 #the columns that will be parsed to extract the fields from the jsons
cols_to_parse = ['device', 'geoNetwork', 'totals', 'trafficSource']

def read_parse_dataframe(file_name):
    #full path for the data file
    path = PATH + file_name
    #read the data file, convert the columns in the list of columns to parse using json loader,
    #convert the `fullVisitorId` field as a string
    data_df = pd.read_csv(path, 
        converters={column: json.loads for column in cols_to_parse}, 
        dtype={'fullVisitorId': 'str'})
    #parse the json-type columns
    for col in cols_to_parse:
        #each column became a dataset, with the columns the fields of the Json type object
        json_col_df = json_normalize(data_df[col])
        json_col_df.columns = [f"{col}_{sub_col}" for sub_col in json_col_df.columns]
        #we drop the object column processed and we add the columns created from the json fields
        data_df = data_df.drop(col, axis=1).merge(json_col_df, right_index=True, left_index=True)
    return data_df
    
def process_date_time(data_df):
    print("process date time ...")
    data_df['date'] = data_df['date'].astype(str)
    data_df["date"] = data_df["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
    data_df["date"] = pd.to_datetime(data_df["date"])   
    data_df["year"] = data_df['date'].dt.year
    data_df["month"] = data_df['date'].dt.month
    data_df["day"] = data_df['date'].dt.day
    data_df["weekday"] = data_df['date'].dt.weekday
    data_df['weekofyear'] = data_df['date'].dt.weekofyear
    data_df['month_unique_user_count'] = data_df.groupby('month')['fullVisitorId'].transform('nunique')
    data_df['day_unique_user_count'] = data_df.groupby('day')['fullVisitorId'].transform('nunique')
    data_df['weekday_unique_user_count'] = data_df.groupby('weekday')['fullVisitorId'].transform('nunique')

    return data_df

def process_format(data_df):
    print("process format ...")
    for col in ['visitNumber', 'totals_hits', 'totals_pageviews']:
        data_df[col] = data_df[col].astype(float)
    data_df['trafficSource_adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    data_df['trafficSource_isTrueDirect'].fillna(False, inplace=True)
    return data_df
    
def process_device(data_df):
    print("process device ...")
    data_df['browser_category'] = data_df['device_browser'] + '_' + data_df['device_deviceCategory']
    data_df['browser_os'] = data_df['device_browser'] + '_' + data_df['device_operatingSystem']
    return data_df

def process_totals(data_df):
    print("process totals ...")
    data_df['visitNumber'] = np.log1p(data_df['visitNumber'])
    data_df['totals_hits'] = np.log1p(data_df['totals_hits'])
    data_df['totals_pageviews'] = np.log1p(data_df['totals_pageviews'].fillna(0))
    data_df['mean_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('mean')
    data_df['sum_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('sum')
    data_df['max_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('max')
    data_df['min_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('min')
    data_df['var_hits_per_day'] = data_df.groupby(['day'])['totals_hits'].transform('var')
    data_df['mean_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('mean')
    data_df['sum_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('sum')
    data_df['max_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('max')
    data_df['min_pageviews_per_day'] = data_df.groupby(['day'])['totals_pageviews'].transform('min')    
    return data_df

def process_geo_network(data_df):
    print("process geo network ...")
    data_df['sum_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('sum')
    data_df['count_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('count')
    data_df['mean_pageviews_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_pageviews'].transform('mean')
    data_df['sum_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('sum')
    data_df['count_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('count')
    data_df['mean_hits_per_network_domain'] = data_df.groupby('geoNetwork_networkDomain')['totals_hits'].transform('mean')
    return data_df

def process_traffic_source(data_df):
    print("process traffic source ...")
    data_df['source_country'] = data_df['trafficSource_source'] + '_' + data_df['geoNetwork_country']
    data_df['campaign_medium'] = data_df['trafficSource_campaign'] + '_' + data_df['trafficSource_medium']
    data_df['medium_hits_mean'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('mean')
    data_df['medium_hits_max'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('max')
    data_df['medium_hits_min'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('min')
    data_df['medium_hits_sum'] = data_df.groupby(['trafficSource_medium'])['totals_hits'].transform('sum')
    return data_df

#Feature processing
## Load data
print('reading train')
train_df = read_parse_dataframe('train.csv')
trn_len = train_df.shape[0]
train_df = process_date_time(train_df)
print('reading test')
test_df = read_parse_dataframe('test.csv')
test_df = process_date_time(test_df)

## Drop columns
cols_to_drop = [col for col in train_df.columns if train_df[col].nunique(dropna=False) == 1]
train_df.drop(cols_to_drop, axis=1, inplace=True)
test_df.drop([col for col in cols_to_drop if col in test_df.columns], axis=1, inplace=True)

###only one not null value
train_df.drop(['trafficSource_campaignCode'], axis=1, inplace=True)

###converting columns format
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].astype(float)
train_df['totals_transactionRevenue'] = train_df['totals_transactionRevenue'].fillna(0)
train_df['totals_transactionRevenue'] = np.log1p(train_df['totals_transactionRevenue'])


## Features engineering
train_df = process_format(train_df)
train_df = process_device(train_df)
train_df = process_totals(train_df)
train_df = process_geo_network(train_df)
train_df = process_traffic_source(train_df)

test_df = process_format(test_df)
test_df = process_device(test_df)
test_df = process_totals(test_df)
test_df = process_geo_network(test_df)
test_df = process_traffic_source(test_df)

## Categorical columns
print("process categorical columns ...")
num_cols = ['month_unique_user_count', 'day_unique_user_count', 'weekday_unique_user_count',
            'visitNumber', 'totals_hits', 'totals_pageviews', 
            'mean_hits_per_day', 'sum_hits_per_day', 'min_hits_per_day', 'max_hits_per_day', 'var_hits_per_day',
            'mean_pageviews_per_day', 'sum_pageviews_per_day', 'min_pageviews_per_day', 'max_pageviews_per_day',
            'sum_pageviews_per_network_domain', 'count_pageviews_per_network_domain', 'mean_pageviews_per_network_domain',
            'sum_hits_per_network_domain', 'count_hits_per_network_domain', 'mean_hits_per_network_domain',
            'medium_hits_mean','medium_hits_min','medium_hits_max','medium_hits_sum']
            
not_used_cols = ["visitNumber", "date", "fullVisitorId", "sessionId", 
        "visitId", "visitStartTime", 'totals_transactionRevenue', 'trafficSource_referralPath']
cat_cols = [col for col in train_df.columns if col not in num_cols and col not in not_used_cols]

merged_df = pd.concat([train_df, test_df])
print('Cat columns : ', len(cat_cols))
ohe_cols = []
for i in cat_cols:
    if len(set(merged_df[i].values)) < 100:
        ohe_cols.append(i)

print('ohe_cols : ', ohe_cols)
print(len(ohe_cols))
merged_df = pd.get_dummies(merged_df, columns = ohe_cols)
train_df = merged_df[:trn_len]
test_df = merged_df[trn_len:]
del merged_df
gc.collect()

for col in cat_cols:
    if col in ohe_cols:
        continue
    #print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))

print('FINAL train shape : ', train_df.shape, ' test shape : ', test_df.shape)
#print(train_df.columns)
train_df = train_df.sort_values('date')
X = train_df.drop(not_used_cols, axis=1)
y = train_df['totals_transactionRevenue']
X_test = test_df.drop([col for col in not_used_cols if col in test_df.columns], axis=1)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn import model_selection, preprocessing, metrics
import matplotlib.pyplot as plt
import seaborn as sns
def categorize_target(x):
    if x < 2:
        return 0
    elif x < 4:
        return 1
    elif x < 6:
        return 2
    elif x < 8:
        return 3
    elif x < 10:
        return 4
    elif x < 12:
        return 5
    elif x < 14:
        return 6
    elif x < 16:
        return 7
    elif x < 18:
        return 8
    elif x < 20:
        return 9
    elif x < 22:
        return 10
    else:
        return 11
from sklearn.model_selection import StratifiedKFold
y_categorized = y.apply(categorize_target)
from sklearn.model_selection import train_test_split

X_train,X_valid, y_train, y_valid = train_test_split(X,y,test_size=0.15, random_state=42)
print(f"X_train shape {X_train.shape}")
print(f"y_train shape {y_train.shape}")
print(f"X_valid shape {X_valid.shape}")
print(f"y_valid shape {y_valid.shape}")
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import itertools
from math import sqrt
from sklearn.metrics import mean_squared_error
from skopt import gp_minimize
import lightgbm as lgb
space  = [Integer(3, 10, name='max_depth'),
          Integer(6, 50, name='num_leaves'),
          Integer(50, 200, name='min_child_samples'),
          Real(1, 400,  name='scale_pos_weight'),
          Real(0.6, 0.9, name='subsample'),
          Real(0.6, 0.9, name='colsample_bytree'),
          Real(10**-3, 10**0, "log-uniform", name='learning_rate'),
          Integer(100, 256, name='max_bin'),
          Real(0.1,0.9, name='feature_fraction'),
          Real(0.5, 1, name='bagging_fraction'),
          Integer(20,100, name='min_data_in_leaf'),
          Integer(0, 6, name='bagging_freq')
         ]
def objective(values):
    
    params = {'max_depth': values[0], 
          'num_leaves': values[1], 
          'min_child_samples': values[2], 
          'scale_pos_weight': values[3],
            'subsample': values[4],
            'colsample_bytree': values[5],
            'learning_rate':values[6],
            'max_bin': values[7],
            'feature_fraction':values[8],
            'bagging_fraction':values[9],
            'min_data_in_leaf':values[10],
            'bagging_freq':values[11],
             'metric':'rmse',
             'nthread': 8,
             'boosting_type': 'gbdt',
             'objective': 'regression',
             'min_child_weight': 0,
             'min_split_gain': 0,
             'subsample_freq': 1}

    print('\nNext set of params.....',params)

    
    early_stopping_rounds = 50
    num_boost_round       = 2000
    xgtrain = lgb.Dataset(X_train, label=y_train)
    xgvalid = lgb.Dataset(X_valid, label=y_valid)
    
    evals_results = {}
    model_lgb     = lgb.train(params,xgtrain,valid_sets=[xgtrain, xgvalid], 
                              valid_names=['train','valid'], 
                               evals_result=evals_results, 
                               num_boost_round=num_boost_round,
                                early_stopping_rounds=early_stopping_rounds,
                               verbose_eval=None, feval=None)
    
    rmse = sqrt(mean_squared_error(y_valid,model_lgb.predict(X_valid)))
    
    print('\nRMSE.....',rmse,".....iter.....", model_lgb.current_iteration())
    
    gc.collect()
    
    return  rmse

res_gp = gp_minimize(objective, space, n_calls=50,
                     random_state=0,n_random_starts=10)

print("Best score=%.4f" % res_gp.fun)
print("""Best parameters:
- max_depth=%d
- num_leaves=%d
- min_child_samples=%d
- scale_pos_weight=%.4f
- subsample=%.4f
- colsample_bytree=%.4f
- learning_rate=%.4f
- max_bin=%d
- feature_fraction=%.2f
- bagging_fraction=%.2f
- min_data_in_leaf=%d
- bagging_freq=%d
    """ % (res_gp.x[0], res_gp.x[1], res_gp.x[2], res_gp.x[3], 
          res_gp.x[4], res_gp.x[5], res_gp.x[6], res_gp.x[7],
          res_gp.x[8], res_gp.x[9], res_gp.x[10], res_gp.x[11]))
"Best score=%.4f" % res_gp.fun
lgb_params1 = {"objective" : "regression", 
               "metric" : "rmse", 
                'max_depth':8, 
               'num_leaves':50,
               'min_child_samples':50,
              'scale_pos_weight':400.0000,
              'subsample':0.6000,
              'colsample_bytree':0.9000,
             'learning_rate':0.0146,
             'max_bin':100,
             'feature_fraction':0.55,
             'bagging_fraction':0.50,
             'min_data_in_leaf':100,
             'bagging_freq':0}
def kfold_lgb_xgb():
    FOLDs = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)

    oof_lgb = np.zeros(len(train_df))
    predictions_lgb = np.zeros(len(test_df))

    features_lgb = list(X.columns)
    feature_importance_df_lgb = pd.DataFrame()

    for fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(X, y_categorized)):
        trn_data = lgb.Dataset(X.iloc[trn_idx], label=y.iloc[trn_idx])
        val_data = lgb.Dataset(X.iloc[val_idx], label=y.iloc[val_idx])

        print("LGB " + str(fold_) + "-" * 50)
        num_round = 20000
        clf = lgb.train(lgb_params1, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 100)
        oof_lgb[val_idx] = clf.predict(X.iloc[val_idx], num_iteration=clf.best_iteration)

        fold_importance_df_lgb = pd.DataFrame()
        fold_importance_df_lgb["feature"] = features_lgb
        fold_importance_df_lgb["importance"] = clf.feature_importance()
        fold_importance_df_lgb["fold"] = fold_ + 1
        feature_importance_df_lgb = pd.concat([feature_importance_df_lgb, fold_importance_df_lgb], axis=0)
        predictions_lgb += clf.predict(X_test, num_iteration=clf.best_iteration) / FOLDs.n_splits

    #lgb.plot_importance(clf, max_num_features=30)    
    cols = feature_importance_df_lgb[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False)[:50].index
    best_features_lgb = feature_importance_df_lgb.loc[feature_importance_df_lgb.feature.isin(cols)]
    plt.figure(figsize=(14,10))
    sns.barplot(x="importance", y="feature", data=best_features_lgb.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig('lgbm_importances.png')
    x = []
    for i in oof_lgb:
        if i < 0:
            x.append(0.0)
        else:
            x.append(i)
    cv_lgb = mean_squared_error(x, y)**0.5
    cv_lgb = str(cv_lgb)
    cv_lgb = cv_lgb[:10]

    pd.DataFrame({'preds': x}).to_csv('lgb_oof_' + cv_lgb + '.csv', index = False)

    print("CV_LGB : ", cv_lgb)
    return cv_lgb, predictions_lgb

cv_lgb, lgb_ans = kfold_lgb_xgb()
x = []
for i in lgb_ans:
    if i < 0:
        x.append(0.0)
    else:
        x.append(i)
np.save('lgb_ans.npy', x)
submission = test_df[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = x
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('lgb_' + cv_lgb + '.csv',index=False)
