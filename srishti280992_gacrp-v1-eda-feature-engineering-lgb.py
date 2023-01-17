# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt
import seaborn as sns
import math

from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from lightgbm.sklearn import LGBMRegressor
import lightgbm as lgb 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

import gc
gc.enable()

#json requirements
import json
from pandas.io.json import json_normalize

from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go

import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
def load_df(csv_path='../input/train.csv', nrows=None):
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

train = load_df("../input/train.csv")
test = load_df("../input/test.csv")
train[:3]
#target variable
#Since we are predicting the natural log of sum of all transactions of the user
#let us sum up the transaction revenue at user level and take a log and then do a scatter plot.
train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
gdf = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(8,6))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12) 
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()
#transforming target column
train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'])
train[:3]
const_cols = [c for c in train.columns if train[c].nunique(dropna=False)==1 ]
const_cols

cols_to_drop = const_cols  #drop constant columns

train = train.drop(cols_to_drop , axis=1)
test = test.drop(cols_to_drop, axis=1)
(set(train.columns).difference(set(test.columns))) #variable which are not common in both test and train
train = train.drop(['trafficSource.campaignCode'] , axis=1)
train.shape #(903653, 34)
test.shape #(804684, 33)
train.head()

percent = (100 * train.isnull().sum() / train.shape[0]).sort_values(ascending=False)

percent[:20]
percent = (100 * test.isnull().sum() / test.shape[0]).sort_values(ascending=False)
percent[:10]
drop_cols_na = ['trafficSource.adContent', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId','trafficSource.referralPath']

train.drop(drop_cols_na, axis=1, inplace=True)
test.drop(drop_cols_na, axis=1, inplace=True)
print(train.shape)
print(test.shape)
test.info()
def find_missing(data):
    # number of missing values
    count_missing = data.isnull().sum().values
    # total records
    total = data.shape[0]
    # percentage of missing
    ratio_missing = count_missing/total
    # return a dataframe to show: feature name, # of missing and % of missing
    return pd.DataFrame(data={'missing_count':count_missing, 'missing_ratio':ratio_missing}, index=data.columns.values)
train_missing = find_missing(train)
test_missing = find_missing(test)
train_missing.reset_index()[['index', 'missing_ratio']]\
    .merge(test_missing.reset_index()[['index', 'missing_ratio']], on='index', how='left')\
    .rename(columns={'index':'columns', 'missing_ratio_x':'train_missing_ratio', 'missing_ratio_y':'test_missing_ratio'})\
    .sort_values(['train_missing_ratio', 'test_missing_ratio'], ascending=False)\
    .query('train_missing_ratio>0')
miss_per = {}
for k, v in dict(train.isna().sum(axis=0)).items():
    if v == 0:
        continue
    miss_per[k] = 100 * float(v) / len(train)
    
import operator 
sorted_x = sorted(miss_per.items(), key=operator.itemgetter(1), reverse=True)
print ("There are " + str(len(miss_per)) + " columns with missing values")

train.info()
# assuming missing values mean it is False, we fill NAs with false
train['trafficSource.isTrueDirect'].fillna(False, inplace=True)
test['trafficSource.isTrueDirect'].fillna(False, inplace=True)
train['trafficSource.keyword'].fillna('nan', inplace=True)
test['trafficSource.keyword'].fillna('nan', inplace=True)
train['totals.bounces'] = train['totals.bounces'].astype('float')
train['totals.bounces'] = train['totals.bounces'].fillna(0)
test['totals.bounces'] = test['totals.bounces'].astype('float')
test['totals.bounces'] = test['totals.bounces'].fillna(0)
train['totals.newVisits'] = train['totals.newVisits'].astype('float')
train['totals.newVisits'] = train['totals.newVisits'].fillna(0)
test['totals.newVisits'] = test['totals.newVisits'].astype('float')
test['totals.newVisits'] = test['totals.newVisits'].fillna(0)
train['totals.pageviews'] = train['totals.pageviews'].astype('float')
train['totals.pageviews'] = train['totals.pageviews'].fillna(0)
test['totals.pageviews'] = test['totals.pageviews'].astype('float')
test['totals.pageviews'] = test['totals.pageviews'].fillna(0)
train.info()
#totals.hits
train['totals.hits'] = train['totals.hits'].astype('float')
test['totals.hits'] = test['totals.hits'].astype('float')
#visitNumber
train['visitNumber'] = train['visitNumber'].astype('float')
test['visitNumber'] = test['visitNumber'].astype('float')
plt.figure(figsize=(10,10))
sns.heatmap(train.corr(),annot=True)
# Date-based features

## on train data
# transforming date
train['date'] = train['date'].astype(str)
train["date"] = train["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
train["date"] = pd.to_datetime(train["date"])  

#extracting features
train['month'] = train['date'].dt.month
train['day'] = train['date'].dt.day
train['weekday'] = train['date'].dt.weekday
train['weekofyear'] = train['date'].dt.weekofyear

## similarly on test
# transforming date
test['date'] = test['date'].astype(str)
test["date"] = test["date"].apply(lambda x : x[:4] + "-" + x[4:6] + "-" + x[6:])
test["date"] = pd.to_datetime(test["date"])  

#extracting features
test['month'] = test['date'].dt.month
test['day'] = test['date'].dt.day
test['weekday'] = test['date'].dt.weekday
test['weekofyear'] = test['date'].dt.weekofyear
# using above features to create user count features

## on train
train['month_unique_user_count'] = train.groupby('month')['fullVisitorId'].transform('nunique')
train['day_unique_user_count'] = train.groupby('day')['fullVisitorId'].transform('nunique')
train['weekday_unique_user_count'] = train.groupby('weekday')['fullVisitorId'].transform('nunique')
train['weekofyear_unique_user_count'] = train.groupby('weekofyear')['fullVisitorId'].transform('nunique')

## on test
test['month_unique_user_count'] = test.groupby('month')['fullVisitorId'].transform('nunique')
test['day_unique_user_count'] = test.groupby('day')['fullVisitorId'].transform('nunique')
test['weekday_unique_user_count'] = test.groupby('weekday')['fullVisitorId'].transform('nunique')
test['weekofyear_unique_user_count'] = test.groupby('weekofyear')['fullVisitorId'].transform('nunique')
train[:3]
# device based features

train['browser_category'] = train['device.browser'] + '_' + train['device.deviceCategory']
train['browser_operatingSystem'] = train['device.browser'] + '_' + train['device.operatingSystem']

test['browser_category'] = test['device.browser'] + '_' + test['device.deviceCategory']
test['browser_operatingSystem'] = test['device.browser'] + '_' + test['device.operatingSystem']

train['source_country'] = train['trafficSource.source'] + '_' + train['geoNetwork.country']
test['source_country'] = test['trafficSource.source'] + '_' + test['geoNetwork.country']
train['totals.hits'] = np.log1p(train['totals.hits'])
test['totals.hits'] = np.log1p(test['totals.hits'].astype(int))

train['sum_pageviews_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
train['count_pageviews_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
train['mean_pageviews_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
train['sum_hits_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
train['count_hits_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
train['mean_hits_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

test['sum_pageviews_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
test['count_pageviews_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
test['mean_pageviews_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
test['sum_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
test['count_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
test['mean_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

train['mean_hits_per_day'] = train.groupby(['day'])['totals.hits'].transform('mean')
train['sum_hits_per_day'] = train.groupby(['day'])['totals.hits'].transform('sum')
test['mean_hits_per_day'] = test.groupby(['day'])['totals.hits'].transform('mean')
test['sum_hits_per_day'] = test.groupby(['day'])['totals.hits'].transform('sum')
train['sum_pageviews_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
train['count_pageviews_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
train['mean_pageviews_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')

train['sum_pageviews_per_region'] = train.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
train['count_pageviews_per_region'] = train.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
train['mean_pageviews_per_region'] = train.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')

test['sum_pageviews_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
test['count_pageviews_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
test['mean_pageviews_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')

test['sum_pageviews_per_region'] = test.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
test['count_pageviews_per_region'] = test.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
test['mean_pageviews_per_region'] = test.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')
train['sum_hits_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
train['count_hits_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
train['mean_hits_per_network_domain'] = train.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

train['sum_hits_per_region'] = train.groupby('geoNetwork.region')['totals.hits'].transform('sum')
train['count_hits_per_region'] = train.groupby('geoNetwork.region')['totals.hits'].transform('count')
train['mean_hits_per_region'] = train.groupby('geoNetwork.region')['totals.hits'].transform('mean')

train['sum_hits_per_country'] = train.groupby('geoNetwork.country')['totals.hits'].transform('sum')
train['count_hits_per_country'] = train.groupby('geoNetwork.country')['totals.hits'].transform('count')
train['mean_hits_per_country'] = train.groupby('geoNetwork.country')['totals.hits'].transform('mean')

test['sum_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
test['count_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
test['mean_hits_per_network_domain'] = test.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

test['sum_hits_per_region'] = test.groupby('geoNetwork.region')['totals.hits'].transform('sum')
test['count_hits_per_region'] = test.groupby('geoNetwork.region')['totals.hits'].transform('count')
test['mean_hits_per_region'] = test.groupby('geoNetwork.region')['totals.hits'].transform('mean')

test['sum_hits_per_country'] = test.groupby('geoNetwork.country')['totals.hits'].transform('sum')
test['count_hits_per_country'] = test.groupby('geoNetwork.country')['totals.hits'].transform('count')
test['mean_hits_per_country'] = test.groupby('geoNetwork.country')['totals.hits'].transform('mean')
train['user_pageviews_sum'] = train.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
train['user_hits_sum'] = train.groupby('fullVisitorId')['totals.hits'].transform('sum')
test['user_pageviews_sum'] = test.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
test['user_hits_sum'] = test.groupby('fullVisitorId')['totals.hits'].transform('sum')

train['user_pageviews_count'] = train.groupby('fullVisitorId')['totals.pageviews'].transform('count')
train['user_hits_count'] = train.groupby('fullVisitorId')['totals.hits'].transform('count')
test['user_pageviews_count'] = test.groupby('fullVisitorId')['totals.pageviews'].transform('count')
test['user_hits_count'] = test.groupby('fullVisitorId')['totals.hits'].transform('count')

train['user_pageviews_sum_to_mean'] = train['user_pageviews_sum'] / train['user_pageviews_sum'].mean()
train['user_hits_sum_to_mean'] = train['user_hits_sum'] / train['user_hits_sum'].mean()
test['user_pageviews_sum_to_mean'] = test['user_pageviews_sum'] / test['user_pageviews_sum'].mean()
test['user_hits_sum_to_mean'] = test['user_hits_sum'] / test['user_hits_sum'].mean()
train['user_pageviews_to_region'] = train['user_pageviews_sum'] / train['mean_pageviews_per_region']
train['user_hits_to_region'] = train['user_hits_sum'] / train['mean_hits_per_region']

test['user_pageviews_to_region'] = test['user_pageviews_sum'] / test['mean_pageviews_per_region']
test['user_hits_to_region'] = test['user_hits_sum'] / test['mean_hits_per_region']
train['trafficSource.keyword'].value_counts()
def kw_category(x):
    x = str(x).lower()
    if x == 'nan':
        return 'nan'
    
    x = ''.join(x.split())
    
    if 'youtube' in x or 'you' in x or 'yo' in x or 'tub' in x:
        return 'youtube'
    elif 'google' in x or 'goo' in x or 'gle' in x:
        return 'google'
    elif 'content' in x or 'targe' in x or 'cont' in x or 'target' in x or 'remarket' in x:
        return 'content_target'
    elif 'auto' in x or 'match' in x or 'tching' in x:
        return 'auto_matching'
    elif 'art' in x or 'entertainment' in x or 'game' in x or 'gamin' in x or 'play' in x:
        return 'art_entertainment_play'
    else:
        return 'other'
    
train['KW_category'] = train['trafficSource.keyword'].apply(kw_category)
test['KW_category'] = test['trafficSource.keyword'].apply(kw_category)    
train['KW_category'].value_counts()
train['trafficSource.keyword'].value_counts()
# generating squared columns
col_x = ['sum_pageviews_per_network_domain', 'sum_hits_per_network_domain',
       'mean_hits_per_network_domain',
       'mean_pageviews_per_network_domain', 'totals.hits',
       'totals.pageviews']

for i in col_x:
    train[i+'_square'] = train[i] ** 2
    test[i+'_square'] = test[i] ** 2
#generating root columns
for i in col_x:
    train[i+'_root'] = train[i] ** 0.5
    test[i+'_root'] = test[i] ** 2
train.info()
train[:3]
train_fullVisitorId = train['fullVisitorId']
train_sessionId = train['sessionId']
train_visitId = train['visitId']

test_fullVisitorId = test['fullVisitorId']
test_id= test_id = test_fullVisitorId.values
test_sessionId = test['sessionId']
test_visitId = test['visitId']

id_cols_to_drop= ["sessionId","fullVisitorId","visitId"]
visit_traits= ["date","visitNumber","visitStartTime"]
base_cat_cols_to_drop = ['trafficSource.keyword']

all_cols_to_drop= id_cols_to_drop + visit_traits + base_cat_cols_to_drop
#dropping these columns from train and test
train.drop(all_cols_to_drop, axis=1, inplace=True)
test.drop(all_cols_to_drop, axis=1, inplace=True)
corr_matrix = train.corr().abs()
corr_matrix
# remove columns due to correlation
# removing one of totals.hits, totals.bounces and totals.pageviews
corr_columns_to_remove= ["totals.pageviews"]

#removing them from train and test
train.drop(corr_columns_to_remove, axis=1, inplace=True)
test.drop(corr_columns_to_remove, axis=1, inplace=True)
train[:3]
train.info()
num_cols = ['totals.bounces',
             'totals.hits',
             'totals.newVisits',
             'month_unique_user_count',
             'day_unique_user_count',
             'weekday_unique_user_count',
             'weekofyear_unique_user_count',
             'sum_pageviews_per_network_domain',
             'count_pageviews_per_network_domain',
             'mean_pageviews_per_network_domain',
             'sum_hits_per_network_domain',
             'count_hits_per_network_domain',
             'mean_hits_per_network_domain',
             'mean_hits_per_day',
             'sum_hits_per_day',
             'sum_pageviews_per_region',
             'count_pageviews_per_region',
             'mean_pageviews_per_region',
             'sum_hits_per_region',
             'count_hits_per_region',
             'mean_hits_per_region',
             'sum_hits_per_country',
             'count_hits_per_country',
             'mean_hits_per_country',
             'user_pageviews_sum',
             'user_hits_sum',
             'user_pageviews_count',
             'user_hits_count',
             'user_pageviews_sum_to_mean',
             'user_hits_sum_to_mean',
             'user_pageviews_to_region',
             'user_hits_to_region',
             'sum_pageviews_per_network_domain_square',
             'sum_hits_per_network_domain_square',
             'mean_hits_per_network_domain_square',
             'mean_pageviews_per_network_domain_square',
             'totals.hits_square',
             'totals.pageviews_square',
             'sum_pageviews_per_network_domain_root',
             'sum_hits_per_network_domain_root',
             'mean_hits_per_network_domain_root',
             'mean_pageviews_per_network_domain_root',
             'totals.hits_root',
             'totals.pageviews_root']

target= ['totals.transactionRevenue']

cat_cols= set(list(train)) - set(target) - set(num_cols)
y = train['totals.transactionRevenue']
train = train.drop('totals.transactionRevenue', axis=1)
train
test
cat_cols
# label encoder
le = preprocessing.LabelEncoder()

for c in cat_cols:
    le.fit(train[c])
    train[c] = le.transform(train[c])
#label encoding for test
for c in cat_cols:
    le.fit(test[c])
    test[c] = le.transform(test[c])
train_x, val_x, train_y, val_y = train_test_split(train, y, test_size=0.25, random_state=20)

lgb_params = {"objective" : "regression", "metric" : "rmse",
              "num_leaves" : 36, "learning_rate" : 0.05, "bagging_fraction" : 0.75, "feature_fraction" : 0.6, "bagging_frequency" : 7}
    
lgb_train = lgb.Dataset(train_x, label=train_y)
lgb_val = lgb.Dataset(val_x, label=val_y)
model = lgb.train(lgb_params, lgb_train, 300, valid_sets=[lgb_val], early_stopping_rounds=50, verbose_eval=100)
preds = model.predict(test, num_iteration=model.best_iteration)
test["PredictedLogRevenue"] = np.expm1(preds)
sub_df = pd.DataFrame({"fullVisitorId":test_id})
sub_df["PredictedLogRevenue"] = np.expm1(preds)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("predictions_test_lgb.csv", index=False)