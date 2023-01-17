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

       'trafficSource.source', 'customDimensions']



def load_df(csv_path):

    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    ans = pd.DataFrame()

    dfs = pd.read_csv(csv_path, sep=',',

            converters={column: json.loads for column in JSON_COLUMNS}, 

            dtype={'fullVisitorId': 'str'}, # Important!!

            chunksize=100000)

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
%%time

train = load_df('../input/train_v2.csv')

test = load_df('../input/test_v2.csv')



print('train date:', min(train['date']), 'to', max(train['date']))

print('test date:', min(test['date']), 'to', max(test['date']))
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

test = reduce_mem_usage(test)
# only train feature

for c in train.columns.values:

    if c not in test.columns.values: print(c)
train['totals.transactionRevenue'].fillna(0, inplace=True)

train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].astype(float))

print(train['totals.transactionRevenue'].describe())
test['totals.transactionRevenue'] = np.nan
all_data = train.append(test, sort=False).reset_index(drop=True)
print(all_data.info())
null_cnt = train.isnull().sum().sort_values()

print(null_cnt[null_cnt > 0])
# fillna object feature

for col in ['trafficSource.keyword',

            'trafficSource.referralPath',

            'trafficSource.adContent']:

    all_data[col].fillna('unknown', inplace=True)



# fillna numeric feature

all_data['totals.pageviews'].fillna(1, inplace=True)

all_data['totals.newVisits'].fillna(0, inplace=True)

all_data['totals.bounces'].fillna(0, inplace=True)

all_data['totals.pageviews'] = all_data['totals.pageviews'].astype(int)

all_data['totals.newVisits'] = all_data['totals.newVisits'].astype(int)

all_data['totals.bounces'] = all_data['totals.bounces'].astype(int)



# fillna boolean feature

all_data['trafficSource.isTrueDirect'].fillna(False, inplace=True)
# drop constant column

constant_column = [col for col in all_data.columns if all_data[col].nunique() == 1]

#for c in constant_column:

#    print(c + ':', train[c].unique())



print('drop columns:', constant_column)

all_data.drop(constant_column, axis=1, inplace=True)
# pickup any visitor

all_data[all_data['fullVisitorId'] == '7813149961404844386'].sort_values(by='visitNumber')[

    ['date','visitId','visitNumber','totals.hits','totals.pageviews']].head(20)
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
plotCategoryRateBar(all_data, train_rev, 'channelGrouping')
print('train all:', len(train))

print('train unique fullVisitorId:', train['fullVisitorId'].nunique())

print('train unique visitId:', train['visitId'].nunique())

print('-' * 30)

print('test all:', len(test))

print('test unique fullVisitorId:', test['fullVisitorId'].nunique())

print('test unique visitId:', test['visitId'].nunique())



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
all_data['_source.country'] = all_data['trafficSource.source'] + '_' + all_data['geoNetwork.country']

all_data['_campaign.medium'] = all_data['trafficSource.campaign'] + '_' + all_data['trafficSource.medium']

all_data['_browser.category'] = all_data['device.browser'] + '_' + all_data['device.deviceCategory']

all_data['_browser.os'] = all_data['device.browser'] + '_' + all_data['device.operatingSystem']
null_cnt = all_data.isnull().sum().sort_values()

print(null_cnt[null_cnt > 0])
all_data.drop(['visitId','visitStartTime'],axis=1,inplace=True)



for i, t in all_data.loc[:, all_data.columns != 'fullVisitorId'].dtypes.iteritems():

    if t == object:

        all_data[i].fillna('unknown', inplace=True)

        all_data[i] = pd.factorize(all_data[i])[0]

        #all_data[i] = all_data[i].astype('category')
all_data.info()
train = all_data[all_data['totals.transactionRevenue'].notnull()]

test = all_data[all_data['totals.transactionRevenue'].isnull()].drop(['totals.transactionRevenue'], axis=1)
test.shape
train_id = train['fullVisitorId']

test_id = test['fullVisitorId']



Y_train_reg = train.pop('totals.transactionRevenue')

#Y_train_cls = (Y_train_reg.fillna(0) > 0).astype(np.uint8)

X_train = train.drop(['fullVisitorId'], axis=1)

X_test  = test.drop(['fullVisitorId'], axis=1)



print(X_train.shape, X_test.shape)
del all_data, train, test, train_rev

gc.collect()



print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],

                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])
from sklearn.model_selection import StratifiedKFold, GroupKFold

from sklearn.metrics import roc_auc_score

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



folds = GroupKFold(n_splits=5)



oof_preds = np.zeros(X_train.shape[0])

sub_preds = np.zeros(X_test.shape[0])

for fold_, (trn_, val_) in enumerate(folds.split(X_train, Y_train_reg, groups=train_id)):

    trn_x, trn_y = X_train.iloc[trn_], Y_train_reg.iloc[trn_]

    val_x, val_y = X_train.iloc[val_], Y_train_reg.iloc[val_]

    

    reg = lgb.LGBMRegressor(**params, n_estimators=3000)

    reg.fit(trn_x, trn_y, eval_set=[(val_x, val_y)], early_stopping_rounds=50, verbose=500)

    

    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)

    sub_preds += reg.predict(X_test, num_iteration=reg.best_iteration_) / folds.n_splits



pred = sub_preds
# Plot feature importance

feature_importance = reg.feature_importances_

feature_importance = 100.0 * (feature_importance / feature_importance.max())

sorted_idx = np.argsort(feature_importance)

sorted_idx = sorted_idx[len(feature_importance) - 30:]

pos = np.arange(sorted_idx.shape[0]) + .5



plt.figure(figsize=(12,8))

plt.barh(pos, feature_importance[sorted_idx], align='center')

plt.yticks(pos, X_train.columns[sorted_idx])

plt.xlabel('Relative Importance')

plt.title('Variable Importance')

plt.show()
submission = pd.DataFrame({'fullVisitorId':test_id, 'PredictedLogRevenue':pred})



submission["PredictedLogRevenue"] = np.expm1(submission["PredictedLogRevenue"])

submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)

submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)



submission_sum = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()

submission_sum["PredictedLogRevenue"] = np.log1p(submission_sum["PredictedLogRevenue"])

submission_sum.to_csv("submission.csv", index=False)

submission_sum.head(20)
submission_sum['PredictedLogRevenue'].describe()