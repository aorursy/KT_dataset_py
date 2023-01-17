import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
print(os.listdir("../input"))

import json
from pandas.io.json import json_normalize

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
def rmse(y_true, y_pred):
    return round(np.sqrt(mean_squared_error(y_true, y_pred)), 5)

def load_df(path, nrows=None):
    json_columns = ['device','geoNetwork','totals','trafficSource']
    
    df = pd.read_csv(path,
                     #make sure that the json in csv will be converted as dict, otherwise, it will be string
                     converters = {json_column: json.loads for json_column in json_columns},

                     #make sure 'fullVisitorId' is string
                     dtype = {'fullVisitorId':'str'},
                     nrows=nrows)

    for json_column in json_columns:
        #conver the dict as a dataframe
        converted_df = json_normalize(df[json_column])

        #format the name, f'{json_column}.{subcolumn}' = '{}.{}'.format(json_column,subcolumn)
        converted_df.columns = [f'{json_column}.{subcolumn}' for subcolumn in converted_df.columns]

        #remove the origin columns that are dict, and add the dataframe made of the new columns, keep both the ids
        df = df.drop(json_column, axis = 1).merge(converted_df, right_index=True, left_index=True)
        
    print(f'file:{path},shape:{df.shape}')
    #output the loaded and processed df
    return df
%%time
train_df = load_df('../input/train.csv')
%%time
test_df = load_df('../input/test.csv')
print('There are 2 columns that are in training set but not in the test set. They are',set(train_df.columns).difference(set(test_df.columns)))
#nunique() will return the unique numbers of a column, by default, dropna = False, but here, we make it True, because we don't want to ignore any null values
unhelpful_columns = []

for column in train_df.columns:
    if train_df[column].nunique(dropna = False) == 1:
        unhelpful_columns.append(column)

print('There are',train_df.shape[1],'columns.',len(unhelpful_columns),'of them have identical values for all rows. And they are as follows:')
print(unhelpful_columns)
#drop the unhelpful columns
unhelpful_columns.append('trafficSource.campaignCode')
for column in unhelpful_columns:
    if column in train_df.columns:
        train_df = train_df.drop(column, axis = 1)
    if column in test_df.columns:
        test_df = test_df.drop(column, axis = 1)
print(train_df.shape,test_df.shape)
#prepare the y_train
y_train = train_df[['totals.transactionRevenue']].fillna(0.0)
y_train['totals.transactionRevenue'] = np.log1p(y_train['totals.transactionRevenue'].astype('float'))
train_df = train_df.drop('totals.transactionRevenue',axis = 1)
print(type(y_train),train_df.shape)
test_df.head()
#process the date and time
train_df.visitStartTime = pd.to_datetime(train_df.visitStartTime, unit='s')
test_df.visitStartTime = pd.to_datetime(test_df.visitStartTime, unit='s')
train_df["date"] = train_df.visitStartTime
test_df["date"] = test_df.visitStartTime
#add weekday, time and day info
for df in [train_df, test_df]:
    df['weekday'] = df['date'].dt.dayofweek.astype(object)
    df['time'] = df['date'].dt.second + df['date'].dt.minute*60 + df['date'].dt.hour*3600
    df['hour'] = df['date'].dt.hour
    df['month'] = df['date'].dt.month   # it must not be included in features during learning!
    df['day'] = df['date'].dt.date  
#add some combined feature
for df in [train_df, test_df]:
    df['source.country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
    df['campaign.medium'] = df['trafficSource.campaign'] + '_' + df['trafficSource.medium']
    df['browser.category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser.os'] = df['device.browser'] + '_' + df['device.operatingSystem']
train_df.head()
for df in [train_df, test_df]:
    df['device_deviceCategory_channelGrouping'] = df['device.deviceCategory'] + "_" + df['channelGrouping']
    df['channelGrouping_browser'] = df['device.browser'] + "_" + df['channelGrouping']
    df['channelGrouping_OS'] = df['device.operatingSystem'] + "_" + df['channelGrouping']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            df[i + "_" + j] = df[i] + "_" + df[j]
    
    df['content.source'] = df['trafficSource.adContent'].astype(str) + "_" + df['source.country']
    df['medium.source'] = df['trafficSource.medium'] + "_" + df['source.country']
test_df.head()
#convert numeric data to floats or boolean
def convert_data_format(df):
    #numeric values
    df['totals.hits'] = df['totals.hits'].astype('float').fillna(0.0)
    df['totals.pageviews'] = df['totals.pageviews'].astype('float').fillna(0.0)
    df['visitNumber'] = df['visitNumber'].astype('float')
    df['trafficSource.adwordsClickInfo.page'] = df['trafficSource.adwordsClickInfo.page'].astype('float').fillna(0.0)
    #boolean values
    df['totals.bounces'] = df['totals.bounces'].fillna(0.0).astype('bool')
    df['totals.newVisits'] = df['totals.newVisits'].fillna(0.0).astype('bool')
    df['trafficSource.isTrueDirect'] = df['trafficSource.isTrueDirect'].fillna('False')
    df['trafficSource.adwordsClickInfo.isVideoAd'] = df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
    return df
train_df = convert_data_format(train_df)
test_df = convert_data_format(test_df)
#get the mean of hits and pageviews,and the max visit number then add these features to each user
for feature in ["totals.hits", "totals.pageviews"]:
    alldata = pd.concat([train_df, test_df], sort=False)
    alldata_id = alldata.groupby("fullVisitorId")[feature].mean()
    train_df["usermean_" + feature] = train_df.fullVisitorId.map(alldata_id)
    train_df["normal" + feature] = (train_df[feature] - min(alldata[feature]))/(max(alldata[feature]) - min(alldata[feature]))
    
    test_df["usermean_" + feature] = test_df.fullVisitorId.map(alldata_id)
    test_df["normal" + feature] = (test_df[feature] - min(alldata[feature]))/(max(alldata[feature]) - min(alldata[feature]))
    
for feature in ["visitNumber"]:
    info = pd.concat([train_df, test_df], sort=False).groupby("fullVisitorId")[feature].max()
    train_df["usermax_" + feature] = train_df.fullVisitorId.map(info)
    test_df["usermax_" + feature] = test_df.fullVisitorId.map(info)
train_df.tail(20)
train_df.shape
#based on the training set analyasis, we trimmed out the rows in test_df that is confirmed zero revenue
bounce_buy_yes = test_df['totals.bounces']!=1.0
bounce_buy_no = test_df['totals.bounces']==1.0

hit_buy_yes = test_df['totals.hits']!=1.0
hit_buy_no = test_df['totals.hits']==1.0

pageview_buy_yes = test_df['totals.pageviews']>1.0
pageview_buy_no = test_df['totals.pageviews']<2.0

# useful_broswer = ['Firefox','Chrome','Edge','Internet Explorer','Safari','Amazon Silk','Opera','Safari (in-app)','Android Webview']
# useful_browser_yes = test_df['device.browser'].isin(useful_broswer)
# useful_browser_no = ~test_df['device.browser'].isin(useful_broswer)

# newly added filters
# useful_OS = ['Chrome OS','Macintosh','Windows','Linux','Android','Windows Phone','iOS']
# useful_OS_yes = test_df['device.operatingSystem'].isin(useful_OS)
# useful_OS_no = ~test_df['device.operatingSystem'].isin(useful_OS)

#nousedful_subContinent = ['Polynesia','Middle Africa','Micronesian Region','Melanesia']
#useful_subContinent_yes = ~test_df['geoNetwork.subContinent'].isin(nousedful_subContinent)
#useful_subContinent_no = test_df['geoNetwork.subContinent'].isin(nousedful_subContinent)

#adwordsClickInfo_page_yes = test_df['trafficSource.adwordsClickInfo.page']<2.0
#adwordsClickInfo_page_no = test_df['trafficSource.adwordsClickInfo.page']>1.0

# adNetworkType_yes = test_df['trafficSource.adwordsClickInfo.adNetworkType'] != 'Search partners'
# adNetworkType_no = test_df['trafficSource.adwordsClickInfo.adNetworkType'] == 'Search partners'

# nouseful_campaign = ['AW - Electronics','All Products','Data Share']
# useful_campaign_yes = ~test_df['trafficSource.campaign'].isin(nouseful_campaign)
# useful_campaign_no = test_df['trafficSource.campaign'].isin(nouseful_campaign)

test_df_trimmed = test_df[bounce_buy_yes & hit_buy_yes & pageview_buy_yes] # & adwordsClickInfo_page_yes & useful_browser_yes & useful_subContinent_yes& useful_OS_yes &
                         # & adNetworkType_yes & useful_campaign_yes]
test_df_dropped = test_df[bounce_buy_no | hit_buy_no | pageview_buy_no] # | adwordsClickInfo_page_no| useful_browser_no | useful_subContinent_no| useful_OS_no |
                         # | adNetworkType_no | useful_campaign_no]
print(test_df.shape[0],test_df_trimmed.shape[0]+ test_df_dropped.shape[0], test_df_trimmed.shape[0], test_df_dropped.shape[0])
test_df.columns
#process the catgory values
cat_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            "trafficSource.adwordsClickInfo.isVideoAd", "trafficSource.isTrueDirect",#below is the newly generated features,
            'weekday','hour','month',
            "source.country","campaign.medium", "browser.category", "browser.os",
            'content.source','medium.source','device_deviceCategory_channelGrouping', 'channelGrouping_browser','channelGrouping_OS',
            'geoNetwork.subContinent_device.deviceCategory',
            'geoNetwork.subContinent_device.operatingSystem',
            'geoNetwork.subContinent_trafficSource.source']
#             'geoNetwork.city_device.browser','geoNetwork.city_device.deviceCategory','geoNetwork.city_device.operatingSystem','geoNetwork.city_trafficSource.source',
#             'geoNetwork.country_device.browser','geoNetwork.country_device.deviceCategory','geoNetwork.country_device.operatingSystem','geoNetwork.country_trafficSource.source'
#             'geoNetwork.country_trafficSource.source',
#             'device_deviceCategory_channelGrouping', 'channelGrouping_browser',
#             'channelGrouping_OS', 'geoNetwork.city_device.browser',
#             'geoNetwork.city_device.deviceCategory',
#             'geoNetwork.city_device.operatingSystem',
#             'geoNetwork.city_trafficSource.source',
#             'geoNetwork.continent_device.browser',
#             'geoNetwork.continent_device.deviceCategory',
#             'geoNetwork.continent_device.operatingSystem',
#             'geoNetwork.continent_trafficSource.source',
#             'geoNetwork.country_device.browser',
#             'geoNetwork.country_device.deviceCategory',
#             'geoNetwork.country_device.operatingSystem',
#             'geoNetwork.country_trafficSource.source',
#             'geoNetwork.metro_device.browser',
#             'geoNetwork.metro_device.deviceCategory',
#             'geoNetwork.metro_device.operatingSystem',
#             'geoNetwork.metro_trafficSource.source',
#             'geoNetwork.networkDomain_device.browser',
#             'geoNetwork.networkDomain_device.deviceCategory',
#             'geoNetwork.networkDomain_device.operatingSystem',
#             'geoNetwork.networkDomain_trafficSource.source',
#             'geoNetwork.region_device.browser',
#             'geoNetwork.region_device.deviceCategory',
#             'geoNetwork.region_device.operatingSystem',
#             'geoNetwork.region_trafficSource.source',
#             'geoNetwork.subContinent_device.browser',
#             'geoNetwork.subContinent_device.deviceCategory',
#             'geoNetwork.subContinent_device.operatingSystem',
#             'geoNetwork.subContinent_trafficSource.source', 'content.source',
#             'medium.source'] 

#category values
for col in cat_cols:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df_trimmed[col].values.astype('str')) + list(test_df_dropped[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df_trimmed[col] = lbl.transform(list(test_df_trimmed[col].values.astype('str')))
    test_df_dropped[col] = lbl.transform(list(test_df_dropped[col].values.astype('str')))
print('Now, the training set has',train_df.shape[0],'rows','the total test set has',test_df.shape[0],'or',test_df_trimmed.shape[0]+ test_df_dropped.shape[0],'rows.',test_df_trimmed.shape[0],'rows are left after trimming; while', test_df_dropped.shape[0],'are dropped. There are',test_df.shape[1],'columns in each dataset.')
train_df.head()
train_df.to_csv('train_df.csv')
test_df_trimmed.to_csv('test_df_trimmed.csv')
test_df_dropped.to_csv('test_df_dropped.csv')
y_train.to_csv('y_train.csv')