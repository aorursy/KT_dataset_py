import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc
import sys
import math

from pandas.io.json import json_normalize
from datetime import datetime
from sklearn import preprocessing

import os
print(os.listdir("../input"))
def load_df(csv_path, nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train = load_df('../input/train.csv')
test = load_df('../input/test.csv')

print('train date:', min(train['date']), 'to', max(train['date']))
print('test date:', min(test['date']), 'to', max(test['date']))
# only train feature
for c in train.columns.values:
    if c not in test.columns.values: print(c)
train['totals.transactionRevenue'].fillna(0, inplace=True)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].astype(float))
print(train['totals.transactionRevenue'].describe())
all_data = train.append(test, sort=False).reset_index(drop=True)
print(all_data.info())
all_data.count()
all_data.isnull().sum()   #null개수 확인
x='channelGrouping'
all_data[x].value_counts()
x='date'
all_data[x].value_counts()
x='fullVisitorId'
all_data[x].value_counts()
x='sessionId'
all_data[x].value_counts()
x='socialEngagementType'
all_data[x].value_counts()
x='visitId'
all_data[x].value_counts()
x='visitNumber'
all_data[x].value_counts()
x='visitStartTime'
all_data[x].value_counts()
x='device.browser'
all_data[x].value_counts()
x='device.browserSize'
all_data[x].value_counts()
x='device.browserVersion'
all_data[x].value_counts()
x='device.deviceCategory'
all_data[x].value_counts()
x='device.flashVersion'
all_data[x].value_counts()
x='device.isMobile'
all_data[x].value_counts()
x='device.language'
all_data[x].value_counts()
x='device.mobileDeviceBranding'
all_data[x].value_counts()
x='device.mobileDeviceInfo'
all_data[x].value_counts()
x='device.mobileDeviceMarketingName'
all_data[x].value_counts()
x='device.mobileDeviceModel'
all_data[x].value_counts()
x='device.mobileInputSelector'
all_data[x].value_counts()
x='device.operatingSystem'
all_data[x].value_counts()
x='device.operatingSystemVersion'
all_data[x].value_counts()
x='device.screenColors'
all_data[x].value_counts()
x='device.screenResolution'
all_data[x].value_counts()
x='geoNetwork.city'
all_data[x].value_counts()
x='geoNetwork.cityId'
all_data[x].value_counts()
x='geoNetwork.continent'
all_data[x].value_counts()
x='geoNetwork.country'
all_data[x].value_counts()
x='geoNetwork.latitude'
all_data[x].value_counts()
x='geoNetwork.longitude'
all_data[x].value_counts()
x='geoNetwork.metro'
all_data[x].value_counts()
x='geoNetwork.networkDomain'
all_data[x].value_counts()
x='geoNetwork.networkLocation'
all_data[x].value_counts()
x='geoNetwork.region'
all_data[x].value_counts()
x='geoNetwork.subContinent'
all_data[x].value_counts()
x='totals.bounces'
all_data[x].value_counts()
x='totals.hits'
all_data[x].value_counts()
x='totals.newVisits'
all_data[x].value_counts()
x='totals.pageviews'
all_data[x].value_counts()
x='totals.transactionRevenue'
all_data[x].value_counts()
x='totals.visits'
all_data[x].value_counts()
x='trafficSource.adContent'
all_data[x].value_counts()
x='trafficSource.adwordsClickInfo.adNetworkType'
all_data[x].value_counts()
x='trafficSource.adwordsClickInfo.criteriaParameters'
all_data[x].value_counts()
x='trafficSource.adwordsClickInfo.gclId'
all_data[x].value_counts()
x='trafficSource.adwordsClickInfo.isVideoAd'
all_data[x].value_counts()
x='trafficSource.adwordsClickInfo.page'
all_data[x].value_counts()
x='trafficSource.adwordsClickInfo.slot'
all_data[x].value_counts()
x='trafficSource.campaign'
all_data[x].value_counts()
x='trafficSource.campaignCode'
all_data[x].value_counts()
x='trafficSource.isTrueDirect'
all_data[x].value_counts()
x='trafficSource.keyword'
all_data[x].value_counts()
x='trafficSource.medium'
all_data[x].value_counts()
x='trafficSource.referralPath'
all_data[x].value_counts()
x='trafficSource.source'
all_data[x].value_counts()
#maxVisitNumber = max(all_data['visitNumber'])
#fvid = all_data[all_data['visitNumber'] == maxVisitNumber]['fullVisitorId']
#all_data[all_data['fullVisitorId'] == fvid.values[0]].sort_values(by='visitNumber')
#all_data['socialEngagementType'].value_counts()
#print(all_data['totals.visits'].value_counts())
#print(all_data['campaignCode'].value_counts())
