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

import os
print(os.listdir("../input"))
gc.enable()

features = ['channelGrouping', 'date', 'fullVisitorId', 'visitId',\
       'visitNumber', 'visitStartTime', 'device.browser',\
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',\
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',\
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',\
       'geoNetwork.subContinent', 'totals.bounces', 'totals.hits',\
       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue',\
       'trafficSource.adContent', 'trafficSource.campaign',\
       'trafficSource.isTrueDirect', 'trafficSource.keyword',\
       'trafficSource.medium', 'trafficSource.referralPath',\
       'trafficSource.source']

def load_df(csv_path):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    ans = pd.DataFrame()
    dfs = pd.read_csv(csv_path, sep=',',
            converters={column: json.loads for column in JSON_COLUMNS}, 
            dtype={'fullVisitorId': 'str'}, # Important!!
            chunksize=200000)
    for df in dfs:
        df.reset_index(drop=True, inplace=True)
        for column in JSON_COLUMNS:
            column_as_df = json_normalize(df[column])
            column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
            df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

        #print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
        use_df = df[features]
        del df
        gc.collect()
        ans = pd.concat([ans, use_df], axis=0).reset_index(drop=True)
        #print(ans.shape)
    return ans
data = [{'id': 1, 'name': {'first': 'Coleen', 'last': 'Volk'}},
        {'name': {'given': 'Mose', 'family': 'Regner'}},
        {'id': 2, 'name': 'Faye Raker'}]

json_normalize(data)
%%time
train = load_df('../input/train.csv')
# test = load_df('../input/test.csv')

print('train date:', min(train['date']), 'to', max(train['date']))
# print('test date:', min(test['date']), 'to', max(test['date']))
train['totals.transactionRevenue'].isnull().value_counts()
# Thanks and credited to https://www.kaggle.com/gemartin
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() 
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() 
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df

train = reduce_mem_usage(train)
# test = reduce_mem_usage(test)
# only train feature
for c in train.columns.values:
    if c not in test.columns.values: print(c)
train['totals.transactionRevenue'].fillna(0, inplace=True)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].astype(float))
print(train['totals.transactionRevenue'].describe())
test['totals.transactionRevenue'] = np.nan
# all_data = train.append(test, sort=False).reset_index(drop=True)
all_data = train.reset_index(drop=True)
print(all_data.info())
null_cnt = train.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])
train['totals.newVisits'].value_counts()
# fillna object feature
for col in ['trafficSource.keyword',
            'trafficSource.referralPath',
            'trafficSource.adContent']:
    all_data[col].fillna('unknown', inplace=True)

# fillna numeric feature
# заполняем в соответствии с продуктовой логикой
all_data['totals.pageviews'].fillna(1, inplace=True) # хотя бы 1 просмотр одной страницы должен быть
all_data['totals.newVisits'].fillna(0, inplace=True) # если NaN - то это не новый пользователь
all_data['totals.bounces'].fillna(0, inplace=True) # если NaN - то это не отскок
all_data['totals.pageviews'] = all_data['totals.pageviews'].astype(int)
all_data['totals.newVisits'] = all_data['totals.newVisits'].astype(int)
all_data['totals.bounces'] = all_data['totals.bounces'].astype(int)

# fillna boolean feature
all_data['trafficSource.isTrueDirect'].fillna(False, inplace=True) # прямой заход на сайт через url
# drop constant column
constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]
#for c in constant_column:
#    print(c + ':', train[c].unique())

print('drop columns:', constant_column)
all_data.drop(constant_column, axis=1, inplace=True)
# pickup any visitor
all_data[all_data['fullVisitorId'] == '7813149961404844386'].sort_values(by='visitNumber')[
    ['date','visitId','visitNumber','totals.hits','totals.pageviews']].head(20)
all_data[all_data['fullVisitorId'] == '7813149961404844386'].sort_values(by='visitNumber').head(20)
print("min_date:", all_data['date'].min(), "max_date:", all_data['date'].max())
train_rev = train[train['totals.transactionRevenue'] > 0].copy()
print(len(train_rev))
train_rev.head()
def plotCategoryRateBar(a, b, colName, topN=np.nan):
    if topN == topN: # isNotNan
        vals = b[colName].value_counts()[:topN]
        subA = a.loc[a[colName].isin(vals.index.values), colName]
        df = pd.DataFrame({'All':subA.value_counts() / len(a), 'Revenue':vals / len(b)})
    else:
        df = pd.DataFrame({'All':a[colName].value_counts() / len(a), 'Revenue':b[colName].value_counts() / len(b)})
    df.sort_values('Revenue').plot.barh(colormap='jet')
print('unique customDimensions count:', train['customDimensions'].nunique())
plotCategoryRateBar(all_data, train_rev, 'customDimensions')
format_str = '%Y%m%d'
all_data['formated_date'] = all_data['date'].apply(lambda x: datetime.strptime(str(x), format_str))
all_data['_year'] = all_data['formated_date'].apply(lambda x:x.year)
all_data['_month'] = all_data['formated_date'].apply(lambda x:x.month)
all_data['_quarterMonth'] = all_data['formated_date'].apply(lambda x:x.day//8)
all_data['_day'] = all_data['formated_date'].apply(lambda x:x.day)
all_data['_weekday'] = all_data['formated_date'].apply(lambda x:x.weekday())

all_data.drop(['date','formated_date'], axis=1, inplace=True)
all_data.head()
plotCategoryRateBar(all_data, train_rev, 'channelGrouping')
print('train all:', len(train))
print('train unique fullVisitorId:', train['fullVisitorId'].nunique())
print('train unique visitId:', train['visitId'].nunique())
print('-' * 30)
# print('test all:', len(test))
# print('test unique fullVisitorId:', test['fullVisitorId'].nunique())
# print('test unique visitId:', test['visitId'].nunique())

#print('common fullVisitorId:', len(pd.merge(train, test, how='inner', on='fullVisitorId'))) # 183434
print(all_data['visitNumber'].value_counts()[:5])
print('-' * 30)
print(all_data['totals.newVisits'].value_counts())
print('-' * 30)
print(all_data['totals.bounces'].value_counts())
#maxVisitNumber = max(all_data['visitNumber'])
#fvid = all_data[all_data['visitNumber'] == maxVisitNumber]['fullVisitorId']
#all_data[all_data['fullVisitorId'] == fvid.values[0]].sort_values(by='visitNumber')
all_data['_visitStartHour'] = all_data['visitStartTime'].apply(
    lambda x: str(datetime.fromtimestamp(x).hour))
all_data.head()
print('unique browser count:', train['device.browser'].nunique())
plotCategoryRateBar(all_data, train_rev, 'device.browser', 10)
pd.crosstab(all_data['device.deviceCategory'], all_data['device.isMobile'], margins=False)
all_data['isMobile'] = True
all_data.loc[all_data['device.deviceCategory'] == 'desktop', 'isMobile'] = False
print('unique operatingSystem count:', train['device.operatingSystem'].nunique())
plotCategoryRateBar(all_data, train_rev, 'device.operatingSystem', 10)
print('unique geoNetwork.city count:', train['geoNetwork.city'].nunique())
plotCategoryRateBar(all_data, train_rev, 'geoNetwork.city', 10)
print('unique geoNetwork.region count:', train['geoNetwork.region'].nunique())
plotCategoryRateBar(all_data, train_rev, 'geoNetwork.region', 10)
all_data['geoNetwork.region'].value_counts()
print('unique geoNetwork.subContinent count:', train['geoNetwork.subContinent'].nunique())
plotCategoryRateBar(all_data, train_rev, 'geoNetwork.subContinent', 10)
print('unique geoNetwork.continent count:', train['geoNetwork.continent'].nunique())
plotCategoryRateBar(all_data, train_rev, 'geoNetwork.continent')
print('unique geoNetwork.metro count:', train['geoNetwork.metro'].nunique())
plotCategoryRateBar(all_data, train_rev, 'geoNetwork.metro', 10)
print('unique geoNetwork.networkDomain count:', train['geoNetwork.networkDomain'].nunique())
plotCategoryRateBar(all_data, train_rev, 'geoNetwork.networkDomain', 10)
print(all_data['totals.hits'].value_counts()[:10])

all_data['totals.hits'] = all_data['totals.hits'].astype(int)
print(all_data['totals.pageviews'].value_counts()[:10])

all_data['totals.pageviews'] = all_data['totals.pageviews'].astype(int)
#print(all_data['totals.visits'].value_counts())
print('unique trafficSource.adContent count:', train['trafficSource.adContent'].nunique())

plotCategoryRateBar(all_data, train_rev, 'trafficSource.adContent', 10)

all_data['_adContentGMC'] = (all_data['trafficSource.adContent'] == 'Google Merchandise Collection').astype(np.uint8)
print('unique trafficSource.campaign count:', train['trafficSource.campaign'].nunique())
plotCategoryRateBar(all_data, train_rev, 'trafficSource.campaign', 10)

all_data['_withCampaign'] = (all_data['trafficSource.campaign'] != '(not set)').astype(np.uint8)
print(all_data['trafficSource.isTrueDirect'].value_counts())
plotCategoryRateBar(all_data, train_rev, 'trafficSource.isTrueDirect')
print('unique trafficSource.keyword count:', train['trafficSource.keyword'].nunique())
plotCategoryRateBar(all_data, train_rev, 'trafficSource.keyword', 10)
print('unique trafficSource.medium count:', train['trafficSource.medium'].nunique())
plotCategoryRateBar(all_data, train_rev, 'trafficSource.medium')
print('unique trafficSource.referralPath count:', train['trafficSource.referralPath'].nunique())
plotCategoryRateBar(all_data, train_rev, 'trafficSource.referralPath', 10)

all_data['_referralRoot'] = (all_data['trafficSource.referralPath'] == '/').astype(np.uint8)
print('unique trafficSource.source count:', train['trafficSource.source'].nunique())
plotCategoryRateBar(all_data, train_rev, 'trafficSource.source', 10)

all_data['_sourceGpmall'] = (all_data['trafficSource.source'] == 'mall.googleplex.com').astype(np.uint8)
_='''
'''
all_data['_meanHitsPerDay'] = all_data.groupby(['_day'])['totals.hits'].transform('mean')
all_data['_meanHitsPerWeekday'] = all_data.groupby(['_weekday'])['totals.hits'].transform('mean')
all_data['_meanHitsPerMonth'] = all_data.groupby(['_month'])['totals.hits'].transform('mean')
all_data['_sumHitsPerDay'] = all_data.groupby(['_day'])['totals.hits'].transform('sum')
all_data['_sumHitsPerWeekday'] = all_data.groupby(['_weekday'])['totals.hits'].transform('sum')
all_data['_sumHitsPerMonth'] = all_data.groupby(['_month'])['totals.hits'].transform('sum')

for feature in ['totals.hits', 'totals.pageviews']:
    info = all_data.groupby('fullVisitorId')[feature].mean()
    all_data['_usermean_' + feature] = all_data.fullVisitorId.map(info)
    
for feature in ['visitNumber']:
    info = all_data.groupby('fullVisitorId')[feature].max()
    all_data['_usermax_' + feature] = all_data.fullVisitorId.map(info)

del info
all_data = all_data.drop(['_meanHitsPerDay', '_meanHitsPerWeekday', '_meanHitsPerMonth', '_sumHitsPerDay', '_sumHitsPerWeekday', 
               '_sumHitsPerMonth', '_usermean_totals.hits', '_usermean_totals.pageviews', '_usermax_visitNumber'], axis=1)
all_data['_source.country'] = all_data['trafficSource.source'] + '_' + all_data['geoNetwork.country']
all_data['_campaign.medium'] = all_data['trafficSource.campaign'] + '_' + all_data['trafficSource.medium']
all_data['_browser.category'] = all_data['device.browser'] + '_' + all_data['device.deviceCategory']
all_data['_browser.os'] = all_data['device.browser'] + '_' + all_data['device.operatingSystem']
all_data = all_data.drop(['_source.country', '_campaign.medium', '_browser.category', '_browser.os'], axis=1)
all_data
print(all_data['device.deviceCategory'].value_counts())
print(all_data['device.operatingSystem'].value_counts())
print(all_data['trafficSource.adContent'].value_counts())
os_set = set(['Windows', 'Macintosh', 'Android', 'iOS', 'Linux', 'Chrome OS'])
all_data['device.operatingSystem.cut'] = all_data['device.operatingSystem'].apply(lambda x: x if x in os_set else 'other')
ad_cnt_set = set(['unknown', 'Google Merchandise Collection', 'Google Online Store'])
all_data['trafficSource.adContent.cut'] = all_data['trafficSource.adContent'].apply(lambda x: x if x in ad_cnt_set else 'other')
all_data = all_data.drop(['device.operatingSystem', 'trafficSource.adContent'], axis=1)
all_data.head()
print(all_data['geoNetwork.city'].value_counts()[:30])
print(all_data['geoNetwork.continent'].value_counts())
print(all_data['geoNetwork.country'].value_counts()[:30])
print(all_data['geoNetwork.metro'].value_counts()[:30])
print(all_data['geoNetwork.networkDomain'].value_counts()[:30])
print(all_data['geoNetwork.region'].value_counts()[:30])
print(all_data['geoNetwork.subContinent'].value_counts()[:30])
city_set = set(all_data['geoNetwork.city'].value_counts()[:30].index)
country_set = set(all_data['geoNetwork.country'].value_counts()[:30].index)
metro_set = set(all_data['geoNetwork.metro'].value_counts()[:30].index)
net_dom_set = set(all_data['geoNetwork.networkDomain'].value_counts()[:30].index)
region_set = set(all_data['geoNetwork.region'].value_counts()[:30].index)
all_data['geoNetwork.city.cut'] = all_data['geoNetwork.city'].apply(lambda x: x if x in city_set else 'other')
all_data['geoNetwork.country.cut'] = all_data['geoNetwork.country'].apply(lambda x: x if x in country_set else 'other')
all_data['geoNetwork.metro.cut'] = all_data['geoNetwork.metro'].apply(lambda x: x if x in metro_set else 'other')
all_data['geoNetwork.networkDomain.cut'] = all_data['geoNetwork.networkDomain'].apply(lambda x: x if x in net_dom_set else 'other')
all_data['geoNetwork.region.cut'] = all_data['geoNetwork.region'].apply(lambda x: x if x in region_set else 'other')
all_data = all_data.drop(['geoNetwork.city', 'geoNetwork.country', 'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region'], axis=1)
print(all_data['trafficSource.medium'].value_counts())
print(all_data['trafficSource.source'].value_counts()[:10])
tr_source_set = set(all_data['trafficSource.source'].value_counts()[:10].index)

all_data['trafficSource.source.cut'] = all_data['trafficSource.source'].apply(lambda x: x if x in tr_source_set else 'other')

all_data = all_data.drop(['trafficSource.source'], axis=1)
all_data = all_data.drop(['trafficSource.keyword', 'trafficSource.campaign', 'trafficSource.referralPath'], axis=1)
train1 = all_data[(all_data._year == 2016) | ((all_data._year == 2017) & (all_data._month <= 2))]
test1 = all_data[(all_data._year == 2017) & (all_data._month >= 3) & (all_data._month <= 4)]

print(train1.shape)
print(test1.shape)
train2 = all_data[((all_data._year == 2016) & (all_data._month >= 11)) | ((all_data._year == 2017) & (all_data._month <= 4))]
test2 = all_data[(all_data._year == 2017) & (all_data._month >= 5) & (all_data._month <= 6)]

print(train2.shape)
print(test2.shape)
train1['sinceLastTime'] = train1['visitStartTime'].max() - train1['visitStartTime']
train2['sinceLastTime'] = train2['visitStartTime'].max() - train2['visitStartTime']
train1.head()
train1['totals.transactionRevenue'] = np.expm1(train1['totals.transactionRevenue'])
test1['totals.transactionRevenue'] = np.expm1(test1['totals.transactionRevenue'])
train2['totals.transactionRevenue'] = np.expm1(train2['totals.transactionRevenue'])
test2['totals.transactionRevenue'] = np.expm1(test2['totals.transactionRevenue'])
num_columns = ['device.isMobile', 'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews',
              'totals.transactionRevenue', 'trafficSource.isTrueDirect', '_visitStartHour', 'isMobile',
              '_adContentGMC', '_withCampaign', '_referralRoot', '_sourceGpmall']
categ_columns = ['channelGrouping', 'device.browser', 'device.deviceCategory', 'geoNetwork.continent',
                                 'geoNetwork.subContinent', 'trafficSource.medium', 'device.operatingSystem.cut',
                                 'trafficSource.adContent.cut', 'geoNetwork.city.cut', 'geoNetwork.country.cut',
                                 'geoNetwork.metro.cut', 'geoNetwork.networkDomain.cut', 'geoNetwork.region.cut',
                                 'trafficSource.source.cut']
time_columns = ['sinceLastTime']
train1_group = train1.groupby(['fullVisitorId'])
train2_group = train2.groupby(['fullVisitorId'])
train1_vis_categ = train1_group[categ_columns].first().reset_index()
train2_vis_categ = train2_group[categ_columns].first().reset_index()
train1_vis_num = train1_group[num_columns].agg(['mean', 'max', 'sum'])
train1_vis_num.columns = ['_'.join(col).strip() for col in train1_vis_num.columns.values]
train1_vis_num = train1_vis_num.reset_index()
train2_vis_num = train2_group[num_columns].agg(['mean', 'max', 'sum'])
train2_vis_num.columns = ['_'.join(col).strip() for col in train2_vis_num.columns.values]
train2_vis_num = train2_vis_num.reset_index()
train1_vis_time = train1_group[time_columns].min().reset_index()
train2_vis_time = train2_group[time_columns].min().reset_index()
train1_vis = train1_vis_categ.merge(train1_vis_num, on='fullVisitorId').merge(train1_vis_time, on='fullVisitorId')
train2_vis = train2_vis_categ.merge(train2_vis_num, on='fullVisitorId').merge(train2_vis_time, on='fullVisitorId')
train1[train1.fullVisitorId == '0002871498069867123']
for fac in ['totals.transactionRevenue_mean', 'totals.transactionRevenue_max', 'totals.transactionRevenue_sum']:
    train1_vis[fac] = np.log1p(train1_vis[fac])
for fac in ['totals.transactionRevenue_mean', 'totals.transactionRevenue_max', 'totals.transactionRevenue_sum']:
    train2_vis[fac] = np.log1p(train2_vis[fac])
train1_vis[train1_vis['totals.transactionRevenue_sum'] > 0]
test1_y = test1.groupby(['fullVisitorId'])['totals.transactionRevenue'].sum().reset_index()
test2_y = test2.groupby(['fullVisitorId'])['totals.transactionRevenue'].sum().reset_index()
test1_y['totals.transactionRevenue'] = np.log1p(test1_y['totals.transactionRevenue'])
test2_y['totals.transactionRevenue'] = np.log1p(test2_y['totals.transactionRevenue'])
print(test1_y[test1_y['totals.transactionRevenue'] > 0].shape)
print(test2_y[test2_y['totals.transactionRevenue'] > 0].shape)
train1_vis = train1_vis.merge(test1_y, on='fullVisitorId', how='left')
train1_vis['target'] = train1_vis['totals.transactionRevenue'].fillna(0)
train1_vis = train1_vis.drop(['totals.transactionRevenue'], axis=1)

train1_vis
train2_vis = train2_vis.merge(test2_y, on='fullVisitorId', how='left')
train2_vis['target'] = train2_vis['totals.transactionRevenue'].fillna(0)
train2_vis = train2_vis.drop(['totals.transactionRevenue'], axis=1)

train2_vis
train1_vis[train1_vis.target > 0]
train1_vis_dumm = pd.get_dummies(train1_vis, columns=['channelGrouping', 'device.browser', 'device.deviceCategory', 'geoNetwork.continent',
                                 'geoNetwork.subContinent', 'trafficSource.medium', 'device.operatingSystem.cut',
                                 'trafficSource.adContent.cut', 'geoNetwork.city.cut', 'geoNetwork.country.cut',
                                 'geoNetwork.metro.cut', 'geoNetwork.networkDomain.cut', 'geoNetwork.region.cut',
                                 'trafficSource.source.cut'])
train2_vis_dumm = pd.get_dummies(train2_vis, columns=['channelGrouping', 'device.browser', 'device.deviceCategory', 'geoNetwork.continent',
                                 'geoNetwork.subContinent', 'trafficSource.medium', 'device.operatingSystem.cut',
                                 'trafficSource.adContent.cut', 'geoNetwork.city.cut', 'geoNetwork.country.cut',
                                 'geoNetwork.metro.cut', 'geoNetwork.networkDomain.cut', 'geoNetwork.region.cut',
                                 'trafficSource.source.cut'])
cols_to_drop = set(train2_vis_dumm.columns) - set(train1_vis_dumm.columns)

train2_vis_dumm = train2_vis_dumm.drop(list(cols_to_drop), axis=1)
cols_to_add = set(train1_vis_dumm.columns) - set(train2_vis_dumm.columns)

for col in cols_to_add:
    train2_vis_dumm[col] = 0
from sklearn import metrics
import lightgbm as lgb
params={'learning_rate': 0.01,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        'random_state':42,
        'bagging_fraction': 0.6,
        'feature_fraction': 0.6
       }

reg = lgb.LGBMRegressor(**params, n_estimators=500)

reg.fit(train1_vis_dumm.drop(['fullVisitorId', 'target'], axis=1), train1_vis_dumm['target'],
        eval_set=[(train2_vis_dumm.drop(['fullVisitorId', 'target'], axis=1), train2_vis_dumm['target'])],
        early_stopping_rounds=50,
        verbose=1)
predicts_train = reg.predict(train1_vis_dumm.drop(['fullVisitorId', 'target'], axis=1))
predicts_valid = reg.predict(train2_vis_dumm.drop(['fullVisitorId', 'target'], axis=1))
print("rmse train:", np.sqrt(metrics.mean_squared_error(predicts_train, train1_vis_dumm['target'])))
print("rmse valid:", np.sqrt(metrics.mean_squared_error(predicts_valid, train2_vis_dumm['target'])))
train1_vis_dumm['predicts'] = predicts_train
train2_vis_dumm['predicts'] = predicts_valid
predicts_train.max()
train2_vis_dumm[train2_vis_dumm.target > 15][['target', 'predicts']]
# Plot feature importance
feature_importance = reg.feature_importances_
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
sorted_idx = sorted_idx[len(feature_importance) - 30:]
pos = np.arange(sorted_idx.shape[0]) + .5

plt.figure(figsize=(12,8))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, train1_vis_dumm.drop(['fullVisitorId', 'target'], axis=1).columns[sorted_idx])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.show()
