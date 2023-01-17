# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import time, datetime, gc, re
from lightgbm import LGBMRegressor
from sklearn.model_selection import StratifiedKFold, KFold, GroupKFold
from sklearn.metrics import mean_squared_error
from collections import Counter

gc.enable()
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str, "visitId":str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str, "visitId":str}, nrows=None)

train_store_1 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_1 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_2 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})

for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(str)

train_exdata = pd.concat([train_store_1, train_store_2], sort=False)
test_exdata = pd.concat([test_store_1, test_store_2], sort=False)

for df in [train, test]:
    df["visitId"] = df["visitId"].apply(lambda x: x.split('.', 1)[0]).astype(str)

# Merge with train/test data
train_new = train.merge(train_exdata, how="left", on="visitId")
test_new = test.merge(test_exdata, how="left", on="visitId")

# Drop Client Id
for df in [train_new, test_new]:
    df.drop("Client Id", 1, inplace=True)

#Cleaning Revenue
for df in [train_new, test_new]:
    df["Revenue"].fillna('$', inplace=True)
    df["Revenue"] = df["Revenue"].apply(lambda x: x.replace('$', '').replace(',', ''))
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Revenue"].fillna(0.0, inplace=True)

#Imputing NaN
for df in [train_new, test_new]:
    df["Sessions"] = df["Sessions"].fillna(0)
    df["Avg. Session Duration"] = df["Avg. Session Duration"].fillna(0)
    df["Bounce Rate"] = df["Bounce Rate"].fillna(0)
    df["Revenue"] = df["Revenue"].fillna(0)
    df["Transactions"] = df["Transactions"].fillna(0)
    df["Goal Conversion Rate"] = df["Goal Conversion Rate"].fillna(0)
    df['trafficSource.adContent'].fillna('N/A', inplace=True)
    df['trafficSource.isTrueDirect'].fillna('N/A', inplace=True)
    df['trafficSource.referralPath'].fillna('N/A', inplace=True)
    df['trafficSource.keyword'].fillna('N/A', inplace=True)
    df['totals.bounces'].fillna(0.0, inplace=True)
    df['totals.newVisits'].fillna(0.0, inplace=True)
    df['totals.pageviews'].fillna(0.0, inplace=True)
    
del train
del test
train = train_new
test = test_new
del train_new
del test_new
gc.collect()
for df in [train, test]:
    df.rename({'fullVisitorId': 'id', 'totals.transactionRevenue': 'target'}, axis = 1, inplace = True)
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['weekday'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['monthday'] = df['date'].dt.day
    df.sort_values(['id', 'date'], ascending = True, inplace = True)
    df['next_session'] = (df['date'] - df.groupby('id', sort = False)['date'].shift(1)).astype(np.int64) // 1e9 // 60 // 60
    df['prev_session'] = (df['date'] - df.groupby('id', sort = False)['date'].shift(-1)).astype(np.int64) // 1e9 // 60 // 60
    df.sort_index(inplace = True)
train['target'].fillna(0, inplace = True)
user_labels = (train.groupby('id', sort = False)['target'].max() > 0).astype(int)
user_sums = np.log1p(np.array(train.groupby('id', sort = False)['target'].sum()).tolist())
user_ids = train['id'].unique()
session_sums = train['target'].copy()
del train['target']
mobile_words = {'android', 'samsung', 'mini', 'iphone', 'in-app', 'playstation',
                  'mozilla', 'chrome', 'blackberry', 'nokia', 'browser', 'amazon',
                  'lunascape', 'netscape', 'konqueror', 'puffin', 'amazon'}

normal_browsers = {'chrome', 'safari', 'firefox', 'internet explorer', 'edge', 'opera',
                  'coc coc', 'maxthon', 'iron'}

key_sources = {'google', 'youtube', 'yahoo', 'facebook', 'reddit', 'bing', 'outlook', 'linkedin',
              'pinterest', 'ask', 'siliconvalley', 'lunametrics', 'amazon', 'mysearch', 'qiita',
              'messenger', 'twitter', 't.co', 'vk.com', 'search', 'edu', 'mail', 'ad', 'golang',
              'direct', 'dealspotr', 'sashihara', 'phandroid', 'baidu', 'mdn', 'duckduckgo', 'seroundtable',
              'metrics', 'sogou', 'businessinsider', 'github', 'gophergala', 'yandex', 'msn', 'dfa',
              'feedly', 'arstechnica', 'squishable', 'flipboard', 't-online.de', 'sm.cn', 'wow', 'baidu',
              'partners'}

def browser_mapping(x):
    if x in normal_browsers:
        return x
    elif any([word in x for word in mobile_words]):
        return 'mobile_browser'
    elif '(not set)' in x:
        return 'nan'
    else:
        return 'others'

def adcontents_mapping(x):
    if  'google' in x:
        return 'google'
    elif '(not set)' in x or 'nan' in x:
        return 'nan'
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'

def source_mapping(x):
    for word in key_sources:
        if word in x:
            return word
    if '(not set)' in x or 'nan' in x:
        return 'nan'
    else:
        return 'others'
    
for df in [train, test]:
    df['device.browser'] = df['device.browser'].astype(str).map(lambda x: browser_mapping(x.lower()))
    df['trafficSource.adContent'] = df['trafficSource.adContent'].astype(str).map(lambda x: browser_mapping(x.lower()))
    df['trafficSource.source'] = df['trafficSource.source'].astype(str).map(lambda x: source_mapping(x.lower()))
    
pairs = [('trafficSource.source', 'geoNetwork.country'), ('trafficSource.campaign', 'trafficSource.medium'),
        ('device.browser', 'device.deviceCategory'), ('device.browser', 'device.operatingSystem'),
        ('device.browser', 'channelGrouping'), ('device.deviceCategory', 'channelGrouping'), 
         ('device.operatingSystem', 'channelGrouping'),
        ('trafficSource.adContent', 'source_country'), ('trafficSource.medium', 'source_country')]

def get_second_part(word):
    return re.sub('.*\.', '', word)

for df in [train, test]:
    for first, second in pairs:
        df[get_second_part(first) + '_' + get_second_part(second)] = df[first] + '_' + df[second]
    for first in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 
              'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for second in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            df[get_second_part(first) + "_" + get_second_part(second)] = df[first] + "_" + df[second]
excluded_cols =  {'date', 'id', 'visitId', 'visitStartTime', 'sessionId'}

cat_cols = [col for col in train.columns if col not in excluded_cols and train[col].dtype == 'object']

for col in cat_cols:
    train[col], indexer = pd.factorize(train[col])
    test[col] = indexer.get_indexer(test[col])
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = df['id'].unique()

    # Get folds
    folds = GroupKFold(n_splits = n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['id'].isin(unique_vis[trn_vis])],
                ids[df['id'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
n_splits = 5
splits = get_folds(df = train, n_splits = n_splits)

train_cols = [col for col in train.columns if col not in excluded_cols]

oof_preds = np.zeros(train.shape[0])
test_preds = np.zeros(test.shape[0])
val_scores = []

for i in range(n_splits):
    tr_idx, val_idx = splits[i]

    print("Fold:", i + 1, end = '. ')
    train_X, train_y = train[train_cols].iloc[tr_idx].values, np.log1p(session_sums[tr_idx])
    val_X, val_y = train[train_cols].iloc[val_idx].values, np.log1p(session_sums[val_idx])
    
    gbm = LGBMRegressor(num_leaves = 31, learning_rate = 0.03, n_estimators = 1000, subsample= .9,
                        colsample_bytree= .9 , random_state = 1)
    gbm.fit(train_X, train_y, eval_set = (val_X, val_y) ,early_stopping_rounds = 150, verbose = False, 
            eval_metric = 'rmse')
    
    val_pred = gbm.predict(val_X)
    oof_preds[val_idx] = val_pred
    val_scores.append(np.sqrt(mean_squared_error(val_pred, val_y)))
    print('Score:', val_scores[-1])
    
    test_pred = gbm.predict(test[train_cols])
    test_pred[test_pred < 0] = 0
    test_preds += np.expm1(test_pred) / n_splits
    
oof_preds[oof_preds < 0] = 0

print(np.sqrt(mean_squared_error(np.log1p(session_sums), oof_preds)))

train['preds'] = np.expm1(oof_preds)
train['log_preds'] = oof_preds
test['preds'] = test_preds
test['log_preds'] = np.log1p(test_preds)
stats = ['max', 'mean', 'median', 'std', 'size', 'sum']

user_train = train[train_cols + ['id']].groupby('id', sort = False).mean()
user_test = test[train_cols + ['id']].groupby('id', sort = False).mean()


train_preds = train.groupby('id', sort = False).agg({'preds': stats, 'log_preds': ['sum']}).fillna(0)
train_preds.columns = ['pred' + '_' + word for word in stats] + ['log_pred_sum']

for col in ['pred_max', 'pred_mean', 'pred_median', 'pred_sum', 'pred_std']:
    train_preds[col] = np.log1p(train_preds[col])

user_train = user_train.merge(train_preds, left_index = True, right_index = True)

###


test_preds = test.groupby('id', sort = False).agg({'preds': stats, 'log_preds': ['sum']}).fillna(0)
test_preds.columns = ['pred' + '_' + word for word in stats] + ['log_pred_sum']

for col in ['pred_max', 'pred_mean', 'pred_median', 'pred_sum', 'pred_std']:
    test_preds[col] = np.log1p(test_preds[col])

user_test = user_test.merge(test_preds, left_index = True, right_index = True)
time_min = train['visitStartTime'].min()
time_max = train['visitStartTime'].max()
for df in [train, test]:
    df['visitStartTime'] -= time_min
    df['visitStartTime'] /= (time_max - time_min)
    
aggregations = ['min', 'max', 'std']

times_train = train.groupby('id', sort = False)['visitStartTime'].agg(aggregations).fillna(0)
times_train.columns = ['times_' + word for word in aggregations]
times_train['times_diff'] = times_train['times_max'] - times_train['times_min']
times_train['times_diff_n'] = times_train['times_diff'] / user_train['pred_size']
times_train.drop(['times_min', 'times_max'], axis = 1, inplace = True)

times_test = test.groupby('id', sort = False)['visitStartTime'].agg(aggregations).fillna(0)
times_test.columns = ['times_' + word for word in aggregations]
times_test['times_diff'] = times_test['times_max'] - times_test['times_min']
times_test['times_diff_n'] = times_test['times_diff'] / user_test['pred_size']
times_test.drop(['times_min', 'times_max'], axis = 1, inplace = True)

train['date'] = train['date'].astype(str).apply(lambda x: datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10])))
test['date'] = test['date'].astype(str).apply(lambda x: datetime.date(int(x[:4]), int(x[5:7]), int(x[8:10])))

def analyze_dates(user):
    features = []
    dates = sorted(user['date'].values)
    n = user.shape[0]
    diff = (dates[-1] - dates[0]).days/360
    features += [diff, diff/n]
    features += [Counter(dates).most_common()[0][1]]
    
    return features

dates_train = np.array(train.groupby('id', sort = False).apply(lambda x: analyze_dates(x)).tolist())
dates_test = np.array(test.groupby('id', sort = False).apply(lambda x: analyze_dates(x)).tolist())
from xgboost import XGBRegressor

xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.03,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

lgb_params = {
    'learning_rate': 0.03,
    'metric': 'rmse',
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 1,
    'num_leaves': 31
}

lgb_params_2 = {
    'learning_rate': 0.03,
    'metric': 'rmse',
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'random_state': 1,
    'num_leaves': 10
}
splits = get_folds(df = user_train.reset_index(), n_splits = n_splits)

oof_preds = {'lgb': np.zeros(user_train.shape[0]), 
             'xgb': np.zeros(user_train.shape[0]), 
             'weighted': np.zeros(user_train.shape[0])}

sub_preds = {'lgb': np.zeros(user_test.shape[0]), 
             'xgb': np.zeros(user_test.shape[0])}

val_scores = {'lgb': [], 'xgb': [], 'weighted': []}

print(' fold |    lgb   |    xgb   | weighted ')
print('---------------------------------------')
for i in range(n_splits):
    tr, val = splits[i]
    train_X, train_y = np.hstack([user_train.iloc[tr], dates_train[tr], times_train.iloc[tr]]), user_sums[tr]
    val_X, val_y = np.hstack([user_train.iloc[val], dates_train[val], times_train.iloc[val]]), user_sums[val]
    
    models = {'lgb': LGBMRegressor(**lgb_params, n_estimators = 1500), 
              'xgb': XGBRegressor(**xgb_params, n_estimators = 1000)}
    for name in ['xgb', 'lgb']:
        models[name].fit(train_X, train_y, eval_set = [(val_X, val_y)],
            early_stopping_rounds = 100, eval_metric = 'rmse', verbose = False)
        val_pred = models[name].predict(val_X)
        oof_preds[name][val] = val_pred
        val_scores[name].append(np.sqrt(mean_squared_error(val_pred, val_y)))
        test_pred = models[name].predict(np.hstack([user_test, dates_test, times_test]))
        test_pred[test_pred < 0] = 0
        sub_preds[name] += test_pred / n_splits
    val_pred = 0.7 * oof_preds['lgb'][val] + 0.3 * oof_preds['xgb'][val]
    oof_preds['weighted'][val] = val_pred
    val_scores['weighted'].append(np.sqrt(mean_squared_error(val_pred, val_y)))
    
    print(' {fold: 3d}  | {lgb: 1.5f} | {xgb: 1.5f} | {w: 1.5f}'\
          .format(fold = i + 1, lgb = val_scores['lgb'][-1], xgb = val_scores['xgb'][-1], w = val_scores['weighted'][-1]))
    
print('---------------------------------------')
cv_scores = {}
for name in ['lgb', 'xgb']:
    oof_preds[name][oof_preds[name] < 0] = 0    
    cv_scores[name] = mean_squared_error(user_sums, oof_preds[name]) ** .5
cv_scores['weighted'] = np.sqrt(mean_squared_error(user_sums, 0.6 * oof_preds['lgb'] + 0.4 * oof_preds['xgb']))
print('  CV  | {lgb: 1.5f} | {xgb: 1.5f} | {w: 1.5f}'\
      .format(lgb = cv_scores['lgb'], xgb = cv_scores['xgb'], w = cv_scores['weighted']))
sub = pd.DataFrame()
sub['fullVisitorId'] = user_test.index
sub['PredictedLogRevenue'] = sub_preds['lgb'] * 0.6 + sub_preds['xgb'] * 0.4
sub.loc[(test.groupby('id', sort = False)['totals.bounces'].min() == 1).values, 'PredictedLogRevenue'] = 0.
sub.to_csv("sub.csv", index = False)
