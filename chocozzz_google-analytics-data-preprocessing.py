import os
print(os.listdir("../input"))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
%matplotlib inline
train = pd.read_csv('../input/train.csv', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/test.csv', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape
def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids
train['page_hit'] = train['totals.pageviews']/(train['totals.hits']+1)
train.loc[(train['page_hit'] > 0.35), 'is.page_hit'] = 1
train.loc[(train['page_hit'] <= 0.35), 'is.page_hit'] = 0
del train['page_hit']

test['page_hit'] = test['totals.pageviews']/(test['totals.hits']+1)
test.loc[(test['page_hit'] > 0.35), 'is.page_hit'] = 1
test.loc[(test['page_hit'] <= 0.35), 'is.page_hit'] = 0
del test['page_hit']
for df in [train, test]:
    df['vis_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['vis_date'].dt.dayofweek
    df['sess_date_hours'] = df['vis_date'].dt.hour
    df['sess_date_dom'] = df['vis_date'].dt.day
    df.sort_values(['fullVisitorId', 'vis_date'], ascending=True, inplace=True)
    df['next_session_1'] = (
        df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    
    df['next_session_2'] = (
        df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60
    
    df['max_visits'] = df['fullVisitorId'].map(
         df[['fullVisitorId', 'visitNumber']].groupby('fullVisitorId')['visitNumber'].max()
     )
    
    df['nb_pageviews'] = df['date'].map(
        df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum()
    )
    
    df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]
from sklearn.model_selection import KFold

def mean_k_fold_encoding(col, alpha):
    target_name = 'totals.transactionRevenue'
    target_mean_global = train[target_name].mean()
    
    nrows_cat = train.groupby(col)[target_name].count()
    target_means_cats = train.groupby(col)[target_name].mean()
    target_means_cats_adj = (target_means_cats*nrows_cat + 
                             target_mean_global*alpha)/(nrows_cat+alpha)
    # Mapping means to test data
    encoded_col_test = test[col].map(target_means_cats_adj)
    #임의로 추가 한 부분
    encoded_col_test.fillna(target_mean_global, inplace=True)
    encoded_col_test.sort_index(inplace=True)

    kfold = KFold(n_splits=5, shuffle=True, random_state=1989)
    parts = []
    for trn_inx, val_idx in kfold.split(train):
        df_for_estimation, df_estimated = train.iloc[trn_inx], train.iloc[val_idx]
        nrows_cat = df_for_estimation.groupby(col)[target_name].count()
        target_means_cats = df_for_estimation.groupby(col)[target_name].mean()

        target_means_cats_adj = (target_means_cats * nrows_cat + 
                                target_mean_global * alpha) / (nrows_cat + alpha)

        encoded_col_train_part = df_estimated[col].map(target_means_cats_adj)
        parts.append(encoded_col_train_part)
        
    encoded_col_train = pd.concat(parts, axis=0)
    encoded_col_train.fillna(target_mean_global, inplace=True)
    encoded_col_train.sort_index(inplace=True)
    
    return encoded_col_train, encoded_col_test
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)

for col in categorical_features:
    temp_encoded_tr, temp_encoded_te = mean_k_fold_encoding(col, 5)
    new_feat_name = 'mean_k_fold_{}'.format(col)
    train[new_feat_name] = temp_encoded_tr.values
    test[new_feat_name] = temp_encoded_te.values
gc.collect()
def frequency_encoding(frame, col):
    freq_encoding = frame.groupby([col]).size()/frame.shape[0] 
    freq_encoding = freq_encoding.reset_index().rename(columns={0:'{}_Frequency'.format(col)})
    return frame.merge(freq_encoding, on=col, how='left')

len_train = train.shape[0]
df_all = pd.concat([train, test])

for col in categorical_features:
    df_all = frequency_encoding(df_all, col)
del df_all['geoNetwork.subContinent_Frequency']
del df_all['geoNetwork.country_Frequency']
del df_all['geoNetwork.region_Frequency']; del df_all['geoNetwork.city_Frequency']
del df_all['device.deviceCategory_Frequency']; del df_all['geoNetwork.continent_Frequency']
del df_all['trafficSource.adContent_Frequency']; 

df_all['sub_net_mm'] = df_all['mean_k_fold_geoNetwork.subContinent'] * df_all['mean_k_fold_geoNetwork.networkDomain']
df_all['region_city_metro_mmm'] = - (df_all['mean_k_fold_geoNetwork.region'] / df_all['mean_k_fold_geoNetwork.city'] * df_all['mean_k_fold_geoNetwork.metro'])
df_all['metro_source_mm'] = df_all['mean_k_fold_geoNetwork.metro'] * df_all['mean_k_fold_trafficSource.source']
df_all['channel_device_mm'] = df_all['mean_k_fold_channelGrouping'] * df_all['mean_k_fold_device.deviceCategory']

del df_all['mean_k_fold_geoNetwork.subContinent']; del df_all['mean_k_fold_geoNetwork.networkDomain'];
del df_all['mean_k_fold_geoNetwork.region']; del df_all['mean_k_fold_geoNetwork.city']; del df_all['mean_k_fold_geoNetwork.metro']
del df_all['mean_k_fold_trafficSource.source']; del df_all['mean_k_fold_channelGrouping']; del df_all['mean_k_fold_device.deviceCategory']
# https://www.kaggle.com/prashantkikani/teach-lightgbm-to-sum-predictions-fe
def browser_mapping(x):
    browsers = ['chrome','safari','firefox','internet explorer','edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x.lower()
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'
    
    
def adcontents_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'
    
def source_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'

df_all['device.browser'] = df_all['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
df_all['trafficSource.adContent'] = df_all['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
df_all['trafficSource.source'] = df_all['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

def process_device(data_df):
    print("process device ...")
    data_df['source.country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '_' + data_df['trafficSource.medium']
    data_df['browser.category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + "_" + data_df['device.operatingSystem']
    return data_df

df_all = process_device(df_all)

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device.deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device.browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device.operatingSystem'] + "_" + data['channelGrouping']
    data['contry_sess_date_hours'] = data['geoNetwork.country'] + "_" +data['sess_date_hours'].astype(str)
    data['contry_sess_date_dom'] = data['geoNetwork.country'] + "_" +data['sess_date_dom'].astype(str)
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            data[i + "_" + j] = data[i] + "_" + data[j]
    
    data['content.source'] = data['trafficSource.adContent'] + "_" + data['source.country']
    data['medium.source'] = data['trafficSource.medium'] + "_" + data['source.country']
    return data

df_all = custom(df_all)
df_all = df_all.drop(categorical_features, axis=1, inplace=False)
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime', 'vis_date', 'nb_sessions'
]

cat_cols = [
    _f for _f in df_all.columns
    if (_f not in excluded_features) & (df_all[_f].dtype == 'object')
]
for f in cat_cols:
    df_all[f], indexer = pd.factorize(df_all[f])
    
del cat_cols
gc.collect();
train = df_all[:len_train]
test = df_all[len_train:]
del df_all
gc.collect()
y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']
y_reg = pd.DataFrame(y_reg)
train.to_csv("preprocessing_train.csv",index=False)
test.to_csv("preprocessing_test.csv",index=False)
y_reg.to_csv("y_reg.csv",index=False)