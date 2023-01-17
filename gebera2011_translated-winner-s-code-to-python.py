from fastai.basics import *

from IPython.core.pylabtools import figsize

import gc, json

from pandas.io.json import json_normalize

from datetime import datetime

import lightgbm as lgb

gc.enable()
def load_tr(csv_path, nrows=None, skiprows=None):

    

    usecols = ['channelGrouping', 'date', 'device',

       'fullVisitorId', 'geoNetwork', 'totals',

       'trafficSource', 'visitId', 'visitNumber', 'visitStartTime' ]

    json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

    ans = pd.DataFrame()

    trs = pd.read_csv(csv_path, 

            sep=',',

            usecols = usecols,

            converters={column: json.loads for column in json_cols}, 

            dtype={'fullVisitorId': 'str',

                  'channelGrouping': 'str',                 

                  'visitId':'int',

                  'visitNumber':'int',

                  'visitStartTime':'int'}, 

            parse_dates=['date'], 

            chunksize=100000,

            nrows=nrows,

            skiprows=skiprows)

    

    for tr in trs:

        tr.reset_index(drop=True, inplace=True)

        for column in json_cols:

            column_as_tr = json_normalize(tr[column])

            column_as_tr.columns = [f"{column}_{subcolumn}" for subcolumn in column_as_tr.columns]

            tr = tr.drop(column, axis=1).merge(column_as_tr, right_index=True, left_index=True)



        print(f"Loaded {os.path.basename(csv_path)}. Shape: {tr.shape}")

        tr_chunk = tr  #[features]

        del tr

        gc.collect()

        ans = pd.concat([ans, tr_chunk], axis=0).reset_index(drop=True)

        #print(ans.shape)

    return ans
PATH=Path('../input')



tr = load_tr(PATH/'train_v2.csv')

print('train date:', min(tr['date']), 'to', max(tr['date']))



te = load_tr(PATH/'test_v2.csv')

print('test date:', min(te['date']), 'to', max(te['date']))



tr0 = pd.concat([tr, te], axis=0).reset_index()



tr = tr0

#tr = tr0.sample(n=11000, random_state=1)
tr["date"] = pd.to_datetime(tr["date"], infer_datetime_format=True, format="%Y%m%d")

tr['totals_hits'] = tr['totals_hits'].astype(float)

tr['totals_pageviews'] = tr['totals_pageviews'].astype(float)

tr['totals_timeOnSite'] = tr['totals_timeOnSite'].astype(float)

tr['totals_newVisits'] = tr['totals_newVisits'].astype(float)

tr['totals_transactions'] = tr['totals_transactions'].astype(float)

tr['device_isMobile'] = tr['device_isMobile'].astype(bool)

tr['trafficSource_isTrueDirect'] = tr['trafficSource_isTrueDirect'].astype(bool)
#replace all empty fields with NaN

Nulls = ['(not set)', 'not available in demo dataset', '(not provided)', 

         'unknown.unknown', '/', 'Not Socially Engaged']

for null in Nulls:    

    tr.replace(null, np.nan, inplace=True)
tr['totals_totalTransactionRevenue'] = tr['totals_totalTransactionRevenue'].astype(float)

tr['totals_totalTransactionRevenue'].fillna(0, inplace=True)

target = tr['totals_totalTransactionRevenue']
from datetime import datetime, timedelta

tr["date"] = pd.to_datetime(tr["date"], infer_datetime_format=True, format="%Y%m%d")

def getTimeFramewithFeatures(tr, k=1):



    tf = tr.loc[(tr['date'] >= min(tr['date']) + timedelta(days=168*(k-1))) 

              & (tr['date'] < min(tr['date']) + timedelta(days=168*k))]



    tf_fvid = set(tr.loc[(tr['date'] >= min(tr['date']) + timedelta(days=168*k + 46 )) 

                       & (tr['date'] < min(tr['date']) + timedelta(days=168*k + 46 + 62))]['fullVisitorId'])



    tf_returned = tf[tf['fullVisitorId'].isin(tf_fvid)]

    

    tf_tst = tr[tr['fullVisitorId'].isin(set(tf_returned['fullVisitorId']))

             & (tr['date'] >= min(tr['date']) + timedelta(days=168*k + 46))

             & (tr['date'] < min(tr['date']) + timedelta(days=168*k + 46 + 62))]

    

    tf_target = tf_tst.groupby('fullVisitorId')[['totals_totalTransactionRevenue']].sum().apply(np.log1p, axis=1).reset_index()

    tf_target['ret'] = 1

    tf_target.rename(columns={'totals_totalTransactionRevenue': 'target'}, inplace=True)

    

    tf_nonret = pd.DataFrame()

    tf_nonret['fullVisitorId'] = list(set(tf['fullVisitorId']) - tf_fvid)    

    tf_nonret['target'] = 0

    tf_nonret['ret'] = 0

    

    tf_target = pd.concat([tf_target, tf_nonret], axis=0).reset_index(drop=True)

    # len(set(tf['fullVisitorId'])), len(set(tf_target['fullVisitorId']))

    tf_maxdate = max(tf['date'])

    tf_mindate = min(tf['date'])



    tf = tf.groupby('fullVisitorId').agg({

            'geoNetwork_networkDomain': {'networkDomain': lambda x: x.dropna().max()},

            'geoNetwork_city': {'city': lambda x: x.dropna().max()},

            'device_operatingSystem': {'operatingSystem': lambda x: x.dropna().max()},

            'geoNetwork_metro': {'metro': lambda x: x.dropna().max()},

            'geoNetwork_region': {'region': lambda x: x.dropna().max()},

            'channelGrouping': {'channelGrouping': lambda x: x.dropna().max()},

            'trafficSource_referralPath': {'referralPath': lambda x: x.dropna().max()},

            'geoNetwork_country': {'country': lambda x: x.dropna().max()},

            'trafficSource_source': {'source': lambda x: x.dropna().max()},

            'trafficSource_medium': {'medium': lambda x: x.dropna().max()},

            'trafficSource_keyword': {'keyword': lambda x: x.dropna().max()},

            'device_browser':  {'browser': lambda x: x.dropna().max()},

            'trafficSource_adwordsClickInfo.gclId': {'gclId': lambda x: x.dropna().max()},

            'device_deviceCategory': {'deviceCategory': lambda x: x.dropna().max()},

            'geoNetwork_continent': {'continent': lambda x: x.dropna().max()},

            'totals_timeOnSite': {'timeOnSite_sum': lambda x: x.dropna().sum(),

                                  'timeOnSite_min': lambda x: x.dropna().min(), 

                                  'timeOnSite_max': lambda x: x.dropna().max(),

                                  'timeOnSite_mean': lambda x: x.dropna().mean()},

            'totals_pageviews': {'pageviews_sum': lambda x: x.dropna().sum(),

                                 'pageviews_min': lambda x: x.dropna().min(), 

                                 'pageviews_max': lambda x: x.dropna().max(),

                                 'pageviews_mean': lambda x: x.dropna().mean()},

            'totals_hits': {'hits_sum': lambda x: x.dropna().sum(), 

                            'hits_min': lambda x: x.dropna().min(), 

                            'hits_max': lambda x: x.dropna().max(), 

                            'hits_mean': lambda x: x.dropna().mean()},

            'visitStartTime': {'visitStartTime_counts': lambda x: x.dropna().count()},

            'totals_sessionQualityDim': {'sessionQualityDim': lambda x: x.dropna().max()},

            'trafficSource_isTrueDirect': {'isTrueDirect': lambda x: x.dropna().max()},

            'totals_newVisits': {'newVisits_max': lambda x: x.dropna().max()},

            'device_isMobile': {'isMobile': lambda x: x.dropna().max()},

            'visitNumber': {'visitNumber_max' : lambda x: x.dropna().max()}, 

            'totals_totalTransactionRevenue':  {'totalTransactionRevenue_sum':  lambda x:x.dropna().sum()},

            'totals_transactions' : {'transactions' : lambda x:x.dropna().sum()},

            'date': {'first_ses_from_the_period_start': lambda x: x.dropna().min() - tf_mindate,

                     'last_ses_from_the_period_end': lambda x: tf_maxdate - x.dropna().max(),

                     'interval_dates': lambda x: x.dropna().max() - x.dropna().min(),

                     'unqiue_date_num': lambda x: len(set(x.dropna())) },            

                    })



    tf.columns = tf.columns.droplevel()



    tf = pd.merge(tf, tf_target, left_on='fullVisitorId', right_on='fullVisitorId')

    return tf
###Getting parts of train-set

print('Get 1st train part')

tr1 = getTimeFramewithFeatures(tr, k=1)

tr1.to_pickle(PATH/'tr1_clean')



print('Get 2st train part')

tr2 = getTimeFramewithFeatures(tr, k=2)

tr2.to_pickle(PATH/'tr2_clean')



print('Get 3st train part')

tr3 = getTimeFramewithFeatures(tr, k=3)

tr3.to_pickle(PATH/'tr3_clean')



print('Get 4st train part')

tr4 = getTimeFramewithFeatures(tr, k=4)

tr4.to_pickle(PATH/'tr4_clean')
### Construction of the test-set (by analogy as train-set)

print('Get test')

tr5 = tr[tr['date'] >= pd.to_datetime(20180501, infer_datetime_format=True, format="%Y%m%d")]

tr5_maxdate = max(tr5['date'])

tr5_mindate = min(tr5['date'])
tr5 = tr5.groupby('fullVisitorId').agg({

            'geoNetwork_networkDomain': {'networkDomain': lambda x: x.dropna().max()},

            'geoNetwork_city': {'city': lambda x: x.dropna().max()},

            'device_operatingSystem': {'operatingSystem': lambda x: x.dropna().max()},

            'geoNetwork_metro': {'metro': lambda x: x.dropna().max()},

            'geoNetwork_region': {'region': lambda x: x.dropna().max()},

            'channelGrouping': {'channelGrouping': lambda x: x.dropna().max()},

            'trafficSource_referralPath': {'referralPath': lambda x: x.dropna().max()},

            'geoNetwork_country': {'country': lambda x: x.dropna().max()},

            'trafficSource_source': {'source': lambda x: x.dropna().max()},

            'trafficSource_medium': {'medium': lambda x: x.dropna().max()},

            'trafficSource_keyword': {'keyword': lambda x: x.dropna().max()},

            'device_browser':  {'browser': lambda x: x.dropna().max()},

            'trafficSource_adwordsClickInfo.gclId': {'gclId': lambda x: x.dropna().max()},

            'device_deviceCategory': {'deviceCategory': lambda x: x.dropna().max()},

            'geoNetwork_continent': {'continent': lambda x: x.dropna().max()},

            'totals_timeOnSite': {'timeOnSite_sum': lambda x: x.dropna().sum(),

                                  'timeOnSite_min': lambda x: x.dropna().min(), 

                                  'timeOnSite_max': lambda x: x.dropna().max(),

                                  'timeOnSite_mean': lambda x: x.dropna().mean()},

            'totals_pageviews': {'pageviews_sum': lambda x: x.dropna().sum(),

                                 'pageviews_min': lambda x: x.dropna().min(), 

                                 'pageviews_max': lambda x: x.dropna().max(),

                                 'pageviews_mean': lambda x: x.dropna().mean()},

            'totals_hits': {'hits_sum': lambda x: x.dropna().sum(), 

                            'hits_min': lambda x: x.dropna().min(), 

                            'hits_max': lambda x: x.dropna().max(), 

                            'hits_mean': lambda x: x.dropna().mean()},

            'visitStartTime': {'visitStartTime_counts': lambda x: x.dropna().count()},

            'totals_sessionQualityDim': {'sessionQualityDim': lambda x: x.dropna().max()},

            'trafficSource_isTrueDirect': {'isTrueDirect': lambda x: x.dropna().max()},

            'totals_newVisits': {'newVisits_max': lambda x: x.dropna().max()},

            'device_isMobile': {'isMobile': lambda x: x.dropna().max()},

            'visitNumber': {'visitNumber_max' : lambda x: x.dropna().max()}, 

            'totals_totalTransactionRevenue':  {'totalTransactionRevenue_sum':  lambda x:x.dropna().sum()},

            'totals_transactions' : {'transactions' : lambda x:x.dropna().sum()},

            'date': {'first_ses_from_the_period_start': lambda x: x.dropna().min() - tf_mindate,

                     'last_ses_from_the_period_end': lambda x: tf_maxdate - x.dropna().max(),

                     'interval_dates': lambda x: x.dropna().max() - x.dropna().min(),

                     'unqiue_date_num': lambda x: len(set(x.dropna())) },

                    })

tr5.columns = tr5.columns.droplevel()

tr5['target'] = np.nan

tr5['ret'] = np.nan

tr5.to_pickle(PATH/'tr5_clean')
train_all = pd.concat([tr1, tr2, tr3, tr4, tr5], axis=0, sort=False).reset_index(drop=True)

train_all['interval_dates'] = train_all['interval_dates'].dt.days

train_all['first_ses_from_the_period_start'] = train_all['first_ses_from_the_period_start'].dt.days

train_all['last_ses_from_the_period_end'] = train_all['last_ses_from_the_period_end'].dt.days

train_all.to_pickle(PATH/'train_and_test_clean')
# change objects to category type

cat_train(train_all)

train = train_all[train_all['target'].notnull()]

test = train_all[train_all['target'].isnull()]
params_lgb2 = {

        "objective" : "binary",

        "metric" : "binary_logloss",

        "max_leaves": 256,

        "num_leaves" : 15,

        "min_child_samples" : 1,

        "learning_rate" : 0.01,

        "bagging_fraction" : 0.9,

        "feature_fraction" : 0.8,

        "bagging_frequency" : 1           

    }
params_lgb3 = {

        "objective" : "regression",

        "metric" : "rmse", 

        "max_leaves": 256,

        "num_leaves" : 9,

        "min_child_samples" : 1,

        "learning_rate" : 0.01,

        "bagging_fraction" : 0.9,

        "feature_fraction" : 0.8,

        "bagging_frequency" : 1      

    }
target_cols = ['target', 'ret', 'fullVisitorId']



dtrain_all = lgb.Dataset(train.drop(target_cols, axis=1), label=train['ret'])



dtrain_ret = lgb.Dataset(train.drop(target_cols, axis=1)[train['ret']==1], 

                         label=train['target'][train['ret']==1])
pr_lgb_sum = 0

print('Training and predictions')

for i in range(10):

    print('Interation number ', i)

    lgb_model1 = lgb.train(params_lgb2, dtrain_all, num_boost_round=1200)

    pr_lgb = lgb_model1.predict(test.drop(target_cols, axis=1))

    

    lgb_model2 = lgb.train(params_lgb3, dtrain_ret, num_boost_round=368)

    pr_lgb_ret = lgb_model2.predict(test.drop(target_cols, axis=1))

    

    pr_lgb_sum = pr_lgb_sum + pr_lgb*pr_lgb_ret



pr_final2 = pr_lgb_sum/10