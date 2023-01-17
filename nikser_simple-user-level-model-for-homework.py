import os
print(os.listdir("../input"))
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
import time
import gc
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

gc.enable()
warnings.simplefilter('ignore')
sns.set()
%matplotlib inline
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields
train_df = pd.read_csv('../input/data-flattened/train-flattened/train-flattened.csv', dtype={'fullVisitorId': 'str'})
test_df = pd.read_csv('../input/data-flattened/test-flattened/test-flattened.csv', dtype={'fullVisitorId': 'str'})
const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False) == 1]
train_df.drop(columns=const_cols + ['trafficSource.campaignCode'], inplace=True)
test_df.drop(columns=const_cols, inplace=True)
# https://www.kaggle.com/prashantkikani/teach-lightgbm-to-sum-predictions-fe
def browser_mapping(x):
    browsers = ['chrome', 'safari', 'firefox', 'internet explorer', 'edge', 'opera', 'coc coc', 'maxthon', 'iron']
    mobile_browsers = ['android', 'samsung', 'mini', 'iphone', 'in-app', 'playstation', 'mozilla', 'chrome',
                       'blackberry', 'nokia', 'browser', 'amazon', 'lunascape', 'netscape', 'konqueror', 'puffin']
    if x in browsers:
        return x
    elif '(not set)' in x:
        return x
    elif sum([(k in x) for k in mobile_browsers]):
        return 'mobile browser'
    else:
        return 'others'

def adcontents_mapping(x):
    if 'google' in x:
        return 'google'
    elif ('placement' in x) or ('placememnt' in x):
        return 'placement'
    elif ('(not set)' in x) or ('nan' in x):
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'

def source_mapping(x):
    sources = ['google', 'youtube', '(not set)', 'nan', 'yahoo', 'facebook', 'reddit', 'bing', 'quora', 'outlook', 'linkedin',
               'pinterest', 'ask', 'siliconvalley', 'lunametrics', 'amazon', 'mysearch', 'qiita', 'messenger', 'twitter',
               't.co', 'vk.com', 'search', 'edu', 'mail', 'ad', 'golang', 'direct', 'dealspotr', 'sashihara', 'phandroid',
               'baidu', 'mdn', 'duckduckgo', 'seroundtable', 'metrics', 'sogou', 'businessinsider', 'github', 'gophergala',
               'yandex', 'msn', 'dfa', 'feedly', 'arstechnica', 'squishable', 'flipboard', 't-online.de', 'sm.cn', 'wow', 
               'partners']
    for s in sources:
        if s in x:
            return s
    return 'others'

def custom(df):
    print('custom..')
    df['source.country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
    df['campaign.medium'] = df['trafficSource.campaign'] + '_' + df['trafficSource.medium']
    df['browser.category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser.os'] = df['device.browser'] + '_' + df['device.operatingSystem']

    df['device_deviceCategory_channelGrouping'] = df['device.deviceCategory'] + "_" + df['channelGrouping']
    df['channelGrouping_browser'] = df['device.browser'] + "_" + df['channelGrouping']
    df['channelGrouping_OS'] = df['device.operatingSystem'] + "_" + df['channelGrouping']
    
    df['content.source'] = df['trafficSource.adContent'] + "_" + df['source.country']
    df['medium.source'] = df['trafficSource.medium'] + "_" + df['source.country']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            df[i + "_" + j] = df[i] + "_" + df[j]
    
    return df

for df in [train_df, test_df]:
    df['device.browser'] = df['device.browser'].map(lambda x: browser_mapping(str(x).lower()))
    df['trafficSource.adContent'] = df['trafficSource.adContent'].map(lambda x: adcontents_mapping(str(x).lower()))
    df['trafficSource.source'] = df['trafficSource.source'].map(lambda x: source_mapping(str(x).lower()))

train_df = custom(train_df)
test_df = custom(test_df)
for df in [train_df, test_df]:
    df['visitStartTime'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['visitStartTime'].dt.dayofweek
    df['sess_date_hours'] = df['visitStartTime'].dt.hour
    
    df['totals.hits/views'] = df['totals.hits'] / (df['totals.pageviews'] + 1)

    df.sort_values(['fullVisitorId', 'visitStartTime'], ascending=True, inplace=True)
    df['next_session_1'] = (
        df['visitStartTime'] - df[['fullVisitorId', 'visitStartTime']].groupby('fullVisitorId')['visitStartTime'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (
        df['visitStartTime'] - df[['fullVisitorId', 'visitStartTime']].groupby('fullVisitorId')['visitStartTime'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60
num_cols = ['visitNumber', 'totals.hits', 'totals.pageviews', 'totals.hits/views', 'next_session_1', 'next_session_2']
excluded_cols = ['date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 'visitId', 'visitStartTime']
cat_cols = [col for col in train_df.columns 
            if (col not in excluded_cols) and (col not in num_cols)]
all_cols = cat_cols + num_cols

for col in cat_cols:
    train_df[col], indexer = pd.factorize(train_df[col].astype(str))
    test_df[col] = indexer.get_indexer(test_df[col].astype(str))
    
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)

train_df["totals.transactionRevenue"].fillna(0, inplace=True)
# https://www.kaggle.com/mukesh62/lgb-fe-groupkfold-cv-xgb
def get_folds(df=None, n_splits=5, seed=42):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    np.random.seed(seed)
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
params = {'learning_rate': 0.01, 
         'objective': 'regression', 
         'metric': 'rmse', 
         'num_leaves': 49, 
         'verbose': 1, 
         'bagging_fraction': 0.94, 
         'feature_fraction': 0.67, 
         'random_state': 42, 
         'max_depth': 14, 
         'random_seed': 42,
         'bagging_frequency': 5, 
         'lambda_l2': 0.2, 
         'lambda_l1': 0.55, 
         'min_child_samples': 130
        }
train_y = train_df["totals.transactionRevenue"]
folds = get_folds(train_df, n_splits=5)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = train_df[all_cols].iloc[trn_], train_y.iloc[trn_]
    val_x, val_y = train_df[all_cols].iloc[val_], train_y.iloc[val_]
    
    reg = lgb.LGBMRegressor(**params, n_estimators=2000)
    reg.fit(trn_x, np.log1p(trn_y), eval_set=[(val_x, np.log1p(val_y))], early_stopping_rounds=100, 
            verbose=100, eval_metric='rmse')
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = all_cols
    imp_df['gain_reg'] = reg.booster_.feature_importance(importance_type='gain')
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    # LightGBM
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    lgb_preds = reg.predict(test_df[all_cols], num_iteration=reg.best_iteration_)
    lgb_preds[lgb_preds < 0] = 0
    
    sub_preds += np.expm1(lgb_preds) / len(folds)
    
print("LGBM session-level error: ", mean_squared_error(np.log1p(train_y.values), oof_reg_preds) ** .5)

oof_pred_df = pd.DataFrame({"fullVisitorId": train_df["fullVisitorId"].values})
oof_pred_df["transactionRevenue"] = train_df["totals.transactionRevenue"].values
oof_pred_df["PredictedRevenue"] = np.expm1(oof_reg_preds)
oof_pred_df = oof_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
oof_err = np.sqrt(mean_squared_error(np.log1p(oof_pred_df["transactionRevenue"].values), 
                                     np.log1p(oof_pred_df["PredictedRevenue"].values)))
print("LGBM user-level error: ", oof_err)
importances['gain_log'] = np.log1p(importances['gain_reg'])
mean_gain = importances[['gain_reg', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain_reg'])

plt.figure(figsize=(8, 15))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
train_df['predictions'] = np.expm1(oof_reg_preds)
test_df['predictions'] = sub_preds
# Mode functions in scipy and pandas are too slow
def find_mode(x):
    mode, mode_cnt = 0, 0
    d = dict()
    for val in x.values:
        current_cnt = d.get(val, 0) + 1
        d[val] = current_cnt
        if current_cnt > mode_cnt:
            mode = val
            mode_cnt = current_cnt
    return mode

# use top_N predictions as features for user-level model
def parse_predictions(df):
    top_N = 20
    res = dict()
    res['pred_first'] = df.predictions.values[0]
    res['pred_last'] = df.predictions.values[-1]
    for i, pred in enumerate(df.predictions.sort_values(ascending=False).head(top_N)):
        res['pred_' + str(i)] = pred
    return res
%%time
# Aggregate data at User level
# mean for numerical features and mode for categorical
trn_data = train_df[all_cols + ['fullVisitorId']].groupby('fullVisitorId')\
    .agg({c: np.mean if c in num_cols else find_mode for c in all_cols})
trn_pred_list = train_df[['fullVisitorId', 'predictions']].groupby('fullVisitorId').apply(parse_predictions)
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_all_predictions.columns
# Create a DataFrame with VisitorId as index
session_pred_cols = trn_all_predictions.columns[:-2]
trn_all_predictions['t_log_mean'] = np.log1p(trn_all_predictions[session_pred_cols].mean(axis=1))
trn_all_predictions['t_log_median'] = np.log1p(trn_all_predictions[session_pred_cols].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[session_pred_cols]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[session_pred_cols].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[session_pred_cols].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions
gc.collect()
full_data.shape
%%time
sub_data = test_df[all_cols + ['fullVisitorId']].groupby('fullVisitorId')\
    .agg({c: np.mean if c in num_cols else find_mode for c in all_cols})
sub_pred_list = test_df[['fullVisitorId', 'predictions']].groupby('fullVisitorId').apply(parse_predictions)
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
for f in session_pred_cols:
    if f not in sub_all_predictions.columns:
        sub_all_predictions[f] = np.nan
sub_all_predictions['t_log_mean'] = np.log1p(sub_all_predictions[session_pred_cols].mean(axis=1))
sub_all_predictions['t_log_median'] = np.log1p(sub_all_predictions[session_pred_cols].median(axis=1))
sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[session_pred_cols]).sum(axis=1)
sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[session_pred_cols].fillna(0).sum(axis=1))
sub_all_predictions['t_nb_sess'] = sub_all_predictions[session_pred_cols].isnull().sum(axis=1)

sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
del sub_data, sub_all_predictions
gc.collect()
sub_full_data.shape
target = train_df[['fullVisitorId', 'totals.transactionRevenue']].groupby('fullVisitorId').sum()
params = {'learning_rate': 0.01, 
         'objective': 'regression', 
         'metric': 'rmse', 
         'num_leaves': 31, 
         'verbose': 1, 
         'bagging_fraction': 0.93, 
         'feature_fraction': 0.57, 
         'random_state': 42, 
         'max_depth': 14, 
         'random_seed': 42,
         'bagging_frequency': 5, 
         'lambda_l2': 0.62, 
         'lambda_l1': 0.07, 
         'min_child_samples': 179
        }
folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

oof_reg_preds = np.zeros(full_data.shape[0])
sub_preds = np.zeros(sub_full_data.shape[0])
vis_importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = full_data.iloc[trn_], target['totals.transactionRevenue'].iloc[trn_]
    val_x, val_y = full_data.iloc[val_], target['totals.transactionRevenue'].iloc[val_]
    
    reg = lgb.LGBMRegressor(**params,n_estimators=2000)
    reg.fit(trn_x, np.log1p(trn_y), eval_set=[(val_x, np.log1p(val_y))], early_stopping_rounds=100, 
            verbose=100, eval_metric='rmse')
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = trn_x.columns
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)
    
    # LightGBM
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    lgb_preds = reg.predict(sub_full_data[full_data.columns], num_iteration=reg.best_iteration_)
    lgb_preds[lgb_preds < 0] = 0
    
    sub_preds += np.expm1(lgb_preds) / len(folds)
    
print("LGBM Result: ", mean_squared_error(np.log1p(target['totals.transactionRevenue']), oof_reg_preds) ** .5)
vis_importances['gain_log'] = np.log1p(vis_importances['gain'])
mean_gain = vis_importances[['gain', 'feature']].groupby('feature').mean()
vis_importances['mean_gain'] = vis_importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 25))
sns.barplot(x='gain_log', y='feature', data=vis_importances.sort_values('mean_gain', ascending=False).iloc[:300])
sub_full_data['PredictedLogRevenue'] = np.log1p(sub_preds)
sub_full_data[['PredictedLogRevenue']].to_csv('submission.csv', index=True)