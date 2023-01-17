import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import gc
import matplotlib.pyplot as plt
import seaborn as sns
dataset = pd.read_csv('../input/CleanedData.csv')
dataset.head()
categorical_features = ['channelGrouping', 'fullVisitorId', 'visitId',
       'device.browser', 'device.deviceCategory',
       'device.operatingSystem', 'geoNetwork.city',
       'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro',
       'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent',
       'trafficSource.adContent', 'trafficSource.campaign',
       'trafficSource.isTrueDirect', 'trafficSource.keyword',
       'trafficSource.medium', 'trafficSource.referralPath',
       'trafficSource.source']

numerical_features = ['visitNumber','totals.bounces', 'totals.hits',
       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue', 'totals.timeOnSite']

for feature in numerical_features:
    dataset[feature] = dataset[feature].fillna(0)
    dataset[feature] = dataset[feature].apply(pd.to_numeric)
for feature in categorical_features:
    dataset[feature] = dataset[feature].astype('category')

dataset['device.isMobile'] = dataset['device.isMobile'].astype('int')
#set(dataset.columns) - set(categorical_features + numerical_features)
dataset['visitStartTime'] = pd.to_datetime(dataset['visitStartTime'],unit='s')
dataset = dataset.drop(columns=['date'])
dataset.dtypes
left_train = dataset[(dataset['visitStartTime'] > pd.to_datetime('2018-05-01')) & (dataset['visitStartTime'] < pd.to_datetime('2018-10-15'))]
train = left_train
train['vt_to_1201'] = (pd.to_datetime('2018-12-01') - train['visitStartTime']).dt.days
train['tt_to_1201'] = (pd.to_datetime('2018-12-01') - train['visitStartTime']).dt.days
train.loc[train['totals.transactionRevenue'] == 0, 'tt_to_1201'] =999
train.head()
temp = train[['fullVisitorId','channelGrouping',
       'device.browser', 'device.deviceCategory',
       'device.operatingSystem', 
       'geoNetwork.continent', 'geoNetwork.country', 
       'geoNetwork.networkDomain', 
       'geoNetwork.subContinent',
       'trafficSource.campaign',
       'trafficSource.medium', 
       'trafficSource.source']]
def mode(df, key_cols, value_col):
    return df.groupby(key_cols + [value_col]).size() \
             .to_frame('count').reset_index() \
             .sort_values('count', ascending=False) \
             .drop_duplicates(subset=key_cols).drop(columns = 'count')

def get_modes(dt, columns):
    a = mode(dt, ['fullVisitorId'], columns[0])
    for i in range(1,len(columns)):
        b = mode(dt, ['fullVisitorId'], columns[i])
        a = pd.merge(a, b, how='left', on=['fullVisitorId', 'fullVisitorId'])
        
    return a
d=get_modes(temp, ['channelGrouping',
       'device.browser', 'device.deviceCategory',
       'device.operatingSystem', 
       'geoNetwork.continent', 'geoNetwork.country', 
       'geoNetwork.networkDomain', 
       'geoNetwork.subContinent',
       'trafficSource.campaign',
       'trafficSource.medium', 
       'trafficSource.source'])
train['fullVisitorId'] = train['fullVisitorId'].astype(float)
train['totals.hits'] = train['totals.hits'].astype(float)
train['totals.bounces'] = train['totals.bounces'].astype(float)
train['totals.pageviews'] = train['totals.pageviews'].astype(float)
train['totals.newVisits'] = train['totals.newVisits'].astype(float)
train['totals.timeOnSite'] = train['totals.timeOnSite'].astype(float)
train['device.isMobile'] = train['device.isMobile'].astype(float)
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].astype(float)
train_data = train.groupby('fullVisitorId')['totals.hits'].mean()
temp = train.groupby('fullVisitorId')['totals.bounces'].mean()
train_data = pd.concat([train_data, temp], axis = 1)
temp = train.groupby('fullVisitorId')['totals.pageviews'].mean()
train_data = pd.concat([train_data, temp], axis = 1)
temp = train.groupby('fullVisitorId')['totals.newVisits'].mean()
train_data = pd.concat([train_data, temp], axis = 1)
temp = train.groupby('fullVisitorId')['totals.timeOnSite'].mean()
train_data = pd.concat([train_data, temp], axis = 1)
temp = train.groupby('fullVisitorId')['device.isMobile'].mean()
train_data = pd.concat([train_data, temp], axis = 1)
temp = train.groupby('fullVisitorId')['totals.transactionRevenue'].sum()
train_data = pd.concat([train_data, temp], axis = 1)
temp = train.groupby('fullVisitorId')['vt_to_1201'].min()
train_data = pd.concat([train_data, temp], axis = 1)
temp = train.groupby('fullVisitorId')['tt_to_1201'].min()
train_data = pd.concat([train_data, temp], axis = 1)
train_data = pd.concat([train_data, train['fullVisitorId'].astype(float).value_counts()], axis = 1)
f = train_data.reset_index()
f['count'] = f['fullVisitorId']
f = f.drop(columns = 'fullVisitorId')
f
d['fullVisitorId'] = d['fullVisitorId'].astype('category')
f['index'] = f['index'].astype('category')
d.head()
f.head()
result = pd.merge(d, f, how='left',left_on='fullVisitorId', right_on='index')
result.dtypes
result.head()
result.to_csv('result2.csv')