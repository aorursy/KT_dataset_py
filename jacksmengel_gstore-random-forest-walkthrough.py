import numpy as np

import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import warnings
warnings.filterwarnings('ignore')

from pandas.io.json import json_normalize

import json

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

from sklearn.preprocessing import Imputer

from sklearn import preprocessing

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
train = pd.read_csv('../input/train.csv',dtype={'fullVisitorId': 'str'})
test = pd.read_csv('../input/test.csv',dtype={'fullVisitorId': 'str'})
print(train.head())
train_device = pd.DataFrame(list(train.device.apply(json.loads)))
train_geoNetwork = pd.DataFrame(list(train.geoNetwork.apply(json.loads)))
train_totals = pd.DataFrame(list(train.totals.apply(json.loads)))
train_trafficSource = pd.DataFrame(list(train.trafficSource.apply(json.loads)))
train_trafficSource_adwordsClickInfo = pd.DataFrame(list(train_trafficSource['adwordsClickInfo'].apply(json.dumps).apply(json.loads)))

test_device = pd.DataFrame(list(test.device.apply(json.loads)))
test_geoNetwork = pd.DataFrame(list(test.geoNetwork.apply(json.loads)))
test_totals = pd.DataFrame(list(test.totals.apply(json.loads)))
test_trafficSource = pd.DataFrame(list(test.trafficSource.apply(json.loads)))
test_trafficSource_adwordsClickInfo = pd.DataFrame(list(test_trafficSource['adwordsClickInfo'].apply(json.dumps).apply(json.loads)))
train = pd.concat(
    [
        train[
            [
                'channelGrouping',
                'date',
                'fullVisitorId',
                'sessionId',
                'socialEngagementType',
                'visitId',
                'visitNumber',
                'visitStartTime'
            ]
        ],
        train_device,
        train_geoNetwork,
        train_totals,
        train_trafficSource,
        train_trafficSource_adwordsClickInfo
    ],
    axis = 1
)

# columns with no data or are JSON columns which have been flattened
train = train.drop(
    [
        'socialEngagementType',
        'browserSize',
        'browserVersion',
        'flashVersion',
        'mobileDeviceBranding',
        'mobileDeviceInfo',
        'mobileDeviceMarketingName',
        'mobileDeviceModel',
        'mobileInputSelector',
        'operatingSystemVersion',
        'screenColors',
        'screenResolution',
        'cityId',
        'latitude',
        'longitude',
        'networkLocation',
        'adNetworkType',
        'criteriaParameters',
        'gclId',
        'isVideoAd',
        'page',
        'slot',
        'targetingCriteria',
        'adwordsClickInfo'
    ],
    axis = 1
)

test = pd.concat(
    [
        test[[
                'channelGrouping',
                'date',
                'fullVisitorId',
                'sessionId',
                'socialEngagementType',
                'visitId',
                'visitNumber',
                'visitStartTime'
        ]],
        test_device,
        test_geoNetwork,
        test_totals,
        test_trafficSource,
        test_trafficSource_adwordsClickInfo
    ], 
    axis = 1
)

# columns with no data or are JSON columns which have been flattened
test = test.drop(
    [
        'socialEngagementType',
        'browserSize',
        'browserVersion',
        'flashVersion',
        'mobileDeviceBranding',
        'mobileDeviceInfo',
        'mobileDeviceMarketingName',
        'mobileDeviceModel',
        'mobileInputSelector',
        'operatingSystemVersion',
        'screenColors',
        'screenResolution',
        'cityId',
        'latitude',
        'longitude',
        'networkLocation',
        'adNetworkType',
        'criteriaParameters',
        'gclId',
        'isVideoAd',
        'page',
        'slot',
        'targetingCriteria',
        'adwordsClickInfo'
    ],
    axis = 1
)

del train_device
del train_geoNetwork
del train_totals
del train_trafficSource
del train_trafficSource_adwordsClickInfo
del test_device
del test_geoNetwork
del test_totals
del test_trafficSource
del test_trafficSource_adwordsClickInfo
train['visitStartTime'].astype(str).astype(int)
train['visitId'].astype(str).astype(int)
train['transactionRevenue'].fillna(value = '0', inplace = True)
train['transactionRevenue'] = train['transactionRevenue'].astype(int)
train['visitNumber'].astype(str).astype(int)
train['hits'] = train['hits'].astype(int)
train['pageviews'].fillna(value = '0', inplace = True)
train['pageviews'] = train['pageviews'].astype(int)

test['visitStartTime'].astype(str).astype(int)
test['visitId'].astype(str).astype(int)
test['visitNumber'].astype(str).astype(int)
test['hits'] = test['hits'].astype(int)
test['pageviews'].fillna(value = '0', inplace = True)
test['pageviews'] = test['pageviews'].astype(int)
unique_vals = train.nunique().sort_values(ascending = False)
unique_vals = unique_vals.to_frame()
unique_vals = unique_vals.reset_index()
unique_vals.columns = ['column', 'cnt']

dtypes = train.dtypes.to_frame()
dtypes = dtypes.reset_index()
dtypes.columns = ['column', 'type']

profile = pd.merge(
    unique_vals,
    dtypes,
    on = 'column',
    how = 'inner'
)

print(profile.loc[profile['type'] == 'object'])
networkDomain_encoder =  preprocessing.LabelEncoder()
keyword_encoder =  preprocessing.LabelEncoder()
referralPath_encoder =  preprocessing.LabelEncoder()
city_encoder =  preprocessing.LabelEncoder()
visitNumber_encoder =  preprocessing.LabelEncoder()
source_encoder =  preprocessing.LabelEncoder()
region_encoder =  preprocessing.LabelEncoder()
date_encoder =  preprocessing.LabelEncoder()
country_encoder =  preprocessing.LabelEncoder()
metro_encoder =  preprocessing.LabelEncoder()
browser_encoder =  preprocessing.LabelEncoder()
adContent_encoder =  preprocessing.LabelEncoder()
subContinent_encoder =  preprocessing.LabelEncoder()
operatingSystem_encoder =  preprocessing.LabelEncoder()
campaign_encoder =  preprocessing.LabelEncoder()

networkDomain_encoder.fit(train['networkDomain'])
train['keyword'].fillna(value = '0', inplace = True)
keyword_encoder.fit(train['keyword'])
train['referralPath'].fillna(value = '0', inplace = True)
referralPath_encoder.fit(train['referralPath'])
city_encoder.fit(train['city'])
visitNumber_encoder.fit(train['visitNumber'])
source_encoder.fit(train['source'])
region_encoder.fit(train['region'])
date_encoder.fit(train['date'])
country_encoder.fit(train['country'])
metro_encoder.fit(train['metro'])
browser_encoder.fit(train['browser'])

train['adContent'].fillna(value = '0', inplace = True)
adContent_encoder.fit(train['adContent'])
subContinent_encoder.fit(train['subContinent'])
operatingSystem_encoder.fit(train['operatingSystem'])
campaign_encoder.fit(train['campaign'])

train['networkDomain_encoder'] = networkDomain_encoder.transform(train['networkDomain'])
train['keyword_encoder'] = keyword_encoder.transform(train['keyword'])
train['referralPath_encoder'] = referralPath_encoder.transform(train['referralPath'])
train['city_encoder'] = city_encoder.transform(train['city'])
train['visitNumber_encoder'] = visitNumber_encoder.transform(train['visitNumber'])
train['source_encoder'] = source_encoder.transform(train['source'])
train['region_encoder'] = region_encoder.transform(train['region'])
train['date_encoder'] = date_encoder.transform(train['date'])
train['country_encoder'] = country_encoder.transform(train['country'])
train['metro_encoder'] = metro_encoder.transform(train['metro'])
train['browser_encoder'] = browser_encoder.transform(train['browser'])
train['adContent_encoder'] = adContent_encoder.transform(train['adContent'])
train['subContinent_encoder'] = subContinent_encoder.transform(train['subContinent'])
train['operatingSystem_encoder'] = operatingSystem_encoder.transform(train['operatingSystem'])
train['campaign_encoder'] = campaign_encoder.transform(train['campaign'])

test_networkDomain_encoder =  preprocessing.LabelEncoder()
test_keyword_encoder =  preprocessing.LabelEncoder()
test_referralPath_encoder =  preprocessing.LabelEncoder()
test_city_encoder =  preprocessing.LabelEncoder()
test_visitNumber_encoder =  preprocessing.LabelEncoder()
test_source_encoder =  preprocessing.LabelEncoder()
test_region_encoder =  preprocessing.LabelEncoder()
test_date_encoder =  preprocessing.LabelEncoder()
test_country_encoder =  preprocessing.LabelEncoder()
test_metro_encoder =  preprocessing.LabelEncoder()
test_browser_encoder =  preprocessing.LabelEncoder()
test_adContent_encoder =  preprocessing.LabelEncoder()
test_subContinent_encoder =  preprocessing.LabelEncoder()
test_operatingSystem_encoder =  preprocessing.LabelEncoder()
test_campaign_encoder =  preprocessing.LabelEncoder()

test['keyword'].fillna(value = '0', inplace = True)
test_keyword_encoder.fit(test['keyword'])

test['referralPath'].fillna(value = '0', inplace = True)
test_referralPath_encoder.fit(test['referralPath'])
test_city_encoder.fit(test['city'])
test_visitNumber_encoder.fit(test['visitNumber'])
test_source_encoder.fit(test['source'])
test_region_encoder.fit(test['region'])
test_date_encoder.fit(test['date'])
test_country_encoder.fit(test['country'])
test_metro_encoder.fit(test['metro'])
test_browser_encoder.fit(test['browser'])
test_networkDomain_encoder.fit(test['networkDomain'])
test['adContent'].fillna(value = '0', inplace = True)
test_adContent_encoder.fit(test['adContent'])
test_subContinent_encoder.fit(test['subContinent'])
test_operatingSystem_encoder.fit(test['operatingSystem'])
test_campaign_encoder.fit(test['campaign'])

test['networkDomain_encoder'] = test_networkDomain_encoder.transform(test['networkDomain'])
test['keyword_encoder'] = test_keyword_encoder.transform(test['keyword'])
test['referralPath_encoder'] = test_referralPath_encoder.transform(test['referralPath'])
test['city_encoder'] = test_city_encoder.transform(test['city'])
test['visitNumber_encoder'] = test_visitNumber_encoder.transform(test['visitNumber'])
test['source_encoder'] = test_source_encoder.transform(test['source'])
test['region_encoder'] = test_region_encoder.transform(test['region'])
test['date_encoder'] = test_date_encoder.transform(test['date'])
test['country_encoder'] = test_country_encoder.transform(test['country'])
test['metro_encoder'] = test_metro_encoder.transform(test['metro'])
test['browser_encoder'] = test_browser_encoder.transform(test['browser'])
test['adContent_encoder'] = test_adContent_encoder.transform(test['adContent'])
test['subContinent_encoder'] = test_subContinent_encoder.transform(test['subContinent'])
test['operatingSystem_encoder'] = test_operatingSystem_encoder.transform(test['operatingSystem'])
test['campaign_encoder'] = test_campaign_encoder.transform(test['campaign'])
train_one_hot = train[
    [
        'channelGrouping',
        'deviceCategory',
        'isMobile',
        'language',
        'continent',
        'medium',
        'newVisits',
        'visits',
        'campaignCode',
        'isTrueDirect',
        'bounces'
    ]
]

train_one_hot = pd.get_dummies(train_one_hot)

train = pd.concat(
    [
        train,
        train_one_hot
    ],
    axis = 1
)

test_one_hot = test[
    [
        'channelGrouping',
        'deviceCategory',
        'isMobile',
        'language',
        'continent',
        'medium',
        'newVisits',
        'visits',
        'isTrueDirect',
        'bounces'
    ]
]

test_one_hot = pd.get_dummies(test_one_hot)

test = pd.concat(
    [
        test,
        test_one_hot
    ],
    axis = 1
)

del train_one_hot
del test_one_hot
train['date'] = pd.to_datetime(train['date'], format = '%Y%m%d')
train['month'] = pd.DatetimeIndex(train['date']).month
train['year'] = pd.DatetimeIndex(train['date']).year
train['day'] = pd.DatetimeIndex(train['date']).day
train['quarter'] = pd.DatetimeIndex(train['date']).quarter
train['weekday'] = pd.DatetimeIndex(train['date']).weekday
train['weekofyear'] = pd.DatetimeIndex(train['date']).weekofyear
train['is_month_start'] = pd.DatetimeIndex(train['date']).is_month_start
train['is_month_end'] = pd.DatetimeIndex(train['date']).is_month_end
train['is_quarter_start'] = pd.DatetimeIndex(train['date']).is_quarter_start
train['is_quarter_end'] = pd.DatetimeIndex(train['date']).is_quarter_end
train['is_year_start'] = pd.DatetimeIndex(train['date']).is_year_start
train['is_year_end'] = pd.DatetimeIndex(train['date']).is_year_end
print(train[['month','day','year','quarter','weekday','weekofyear','date']].head())

test['date'] = pd.to_datetime(test['date'], format = '%Y%m%d')
test['month'] = pd.DatetimeIndex(test['date']).month
test['year'] = pd.DatetimeIndex(test['date']).year
test['day'] = pd.DatetimeIndex(test['date']).day
test['quarter'] = pd.DatetimeIndex(test['date']).quarter
test['weekday'] = pd.DatetimeIndex(test['date']).weekday
test['weekofyear'] = pd.DatetimeIndex(test['date']).weekofyear
test['is_month_start'] = pd.DatetimeIndex(test['date']).is_month_start
test['is_month_end'] = pd.DatetimeIndex(test['date']).is_month_end
test['is_quarter_start'] = pd.DatetimeIndex(test['date']).is_quarter_start
test['is_quarter_end'] = pd.DatetimeIndex(test['date']).is_quarter_end
test['is_year_start'] = pd.DatetimeIndex(test['date']).is_year_start
test['is_year_end'] = pd.DatetimeIndex(test['date']).is_year_end
print(test[['month','day','year','quarter','weekday','weekofyear','date']].head())
train['visitStartTime'] = pd.to_datetime(train['visitStartTime'], unit = 's')
train['hour'] = pd.DatetimeIndex(train['visitStartTime']).hour
train['minute'] = pd.DatetimeIndex(train['visitStartTime']).minute
print(train[['visitStartTime','hour','minute']].head())

test['visitStartTime'] = pd.to_datetime(test['visitStartTime'], unit = 's')
test['hour'] = pd.DatetimeIndex(test['visitStartTime']).hour
test['minute'] = pd.DatetimeIndex(test['visitStartTime']).minute
print(test[['visitStartTime','hour','minute']].head())
train_staging = train.select_dtypes(exclude = 'object')
train_staging = train_staging.select_dtypes(exclude = 'datetime')
train_staging = train_staging.select_dtypes(exclude = 'bool')

print(train_staging.dtypes)

test_staging = test.select_dtypes(exclude = 'object')
test_staging = test_staging.select_dtypes(exclude = 'datetime')
test_staging = test_staging.select_dtypes(exclude = 'bool')

print(test_staging.dtypes)
train_staging_columns = train_staging.columns

from sklearn.preprocessing import Imputer
imputer = Imputer(strategy = 'mean')
train_staging = imputer.fit_transform(train_staging)
train_staging = pd.DataFrame(
    data = train_staging,
    columns = train_staging_columns
)
print(train_staging.isna().any())

test_staging_columns = test_staging.columns

test_staging = imputer.fit_transform(test_staging)
test_staging = pd.DataFrame(
    data = test_staging,
    columns = test_staging_columns
)
print(test_staging.isna().any())
train_staging, test_staging = train_staging.align(test_staging, join = 'inner', axis = 1)
train_staging['transactionRevenue'] = train['transactionRevenue']
test_staging['fullVisitorId'] = test['fullVisitorId']
train_staging['fullVisitorId'] = train['fullVisitorId']
print(train_staging.head())
train_agg = train_staging \
    .groupby(['fullVisitorId']) \
    .agg(['count','mean','min','max','sum']) \
    .reset_index()

test_agg = test_staging \
    .groupby(['fullVisitorId']) \
    .agg(['count','mean','min','max','sum']) \
    .reset_index()
columns_train = ['fullVisitorId']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in train_agg.columns.levels[0]:
    if var != 'fullVisitorId':
        for stat in train_agg.columns.levels[1][:-1]:
            columns_train.append('%s_%s' % (var, stat))

train_agg.columns = columns_train

columns_test = ['fullVisitorId']

# Convert multi-level index from .agg() into clean columns
# borrowing from: https://www.kaggle.com/willkoehrsen/introduction-to-manual-feature-engineering
for var in test_agg.columns.levels[0]:
    if var != 'fullVisitorId':
        for stat in test_agg.columns.levels[1][:-1]:
            columns_test.append('%s_%s' % (var, stat))

test_agg.columns = columns_test
del train_staging
del train

del test_staging
del test
print(train_agg.dtypes)
import math

def create_target(rev):
    if rev == 0:
        return 0
    else:
        return math.log(rev)

train_agg['TARGET'] = train_agg['transactionRevenue_sum'].apply(create_target)

train_agg = train_agg.drop(
    [
        'transactionRevenue_count',
        'transactionRevenue_mean',
        'transactionRevenue_min',
        'transactionRevenue_max',
        'transactionRevenue_sum'
    ],
    axis = 1
)
train_agg_corr = train_agg.corr()
print(train_agg_corr['TARGET'].sort_values(ascending = False))
from sklearn.metrics import mean_squared_error
from math import sqrt

id_train = train_agg['fullVisitorId']
x = train_agg.drop(['TARGET','fullVisitorId'], axis = 1)
y = train_agg['TARGET']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 0)

model = RandomForestRegressor()
model.fit(x_train, y_train)
predictions = model.predict(x_test)

rms = sqrt(mean_squared_error(y_test, predictions))

print('RMSE train:', rms)
importances = model.feature_importances_
importances_df = pd.DataFrame(
    data = {'column' : x.columns, 'importance' : importances}
)

importances_df = importances_df.sort_values(by = 'importance', ascending = False)

importances_df['weighted'] = importances_df['importance'] / importances_df['importance'].sum()

plt.figure()
plt.title('Feature Importances')
plt.barh(
    importances_df['column'].head(15),
    importances_df['weighted'].head(15)
)
plt.show()
del train_agg
del Imputer
del RandomForestRegressor
del adContent_encoder
del auc
del browser_encoder
del campaign_encoder 
del city_encoder 
del columns_test 
del columns_train
del country_encoder, create_target, date_encoder, imputer, json, json_normalize, keyword_encoder, math, metro_encoder, networkDomain_encoder, operatingSystem_encoder, parameters, plt, preprocessing
del referralPath_encoder
del region_encoder
del sns
del source_encoder
del stat
del subContinent_encoder
del test_adContent_encoder
del test_browser_encoder
del test_campaign_encoder
del test_city_encoder
del test_country_encoder
del test_date_encoder
del test_keyword_encoder
del test_metro_encoder
del test_networkDomain_encoder 
del test_operatingSystem_encoder
del test_referralPath_encoder
del test_region_encoder
del test_source_encoder
del test_staging_columns
del test_subContinent_encoder
del test_visitNumber_encoder
del train_staging_columns
del train_test_split
del var
del visitNumber_encoder
del warnings
#import lightgbm as lightgbm
#from sklearn.model_selection import train_test_split
#from math import sqrt
#from sklearn.metrics import mean_squared_error

#x_train = lightgbm.Dataset(x_train)
#y_train = lightgbm.Dataset(y_train)

#parameters = {
#    'num_leaves':31,
#    'colsample_bytree' : .9,
#    'metric':'l2_root',
#    'learning_rate':0.03,
#    'subsample' : 0.9, 
#    'random_state' : 1,
#    'n_estimators': 1000
#}

#lgbm = lightgbm.train(
#    parameters,
#    x_train,
#    y_train
#)

#p = lgbm.predict(x_test)

#rms = sqrt(mean_squared_error(y_test, p))

#print('LGBM RMSE train:', rms)
predictions_test = model.predict(test_agg.drop(['fullVisitorId'], axis = 1))

submission = pd.DataFrame({
    "fullVisitorId": test_agg['fullVisitorId'].astype(str),
    "PredictedLogRevenue": predictions_test
    })

submission['fullVisitorId'] = submission['fullVisitorId'].astype(str)

import csv

submission.to_csv('submission_rf.csv', quoting=csv.QUOTE_NONNUMERIC, index = False)
#predictions_test_lgbm = lgbm.predict(test_agg.drop(['fullVisitorId'], axis = 1))

#submission_lgbm = pd.DataFrame({
#    "fullVisitorId": test_agg['fullVisitorId'].astype(str),
#    "PredictedLogRevenue": predictions_test_lgbm
#    })

#submission_lgbm['fullVisitorId'] = submission_lgbm['fullVisitorId'].astype(str)

#import csv

#submission_lgbm.to_csv('submission_lgbm.csv', quoting=csv.QUOTE_NONNUMERIC, index = False)