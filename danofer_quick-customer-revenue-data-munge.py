import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
%matplotlib inline

import numpy as np
import pandas as pd
import json
# import missingno as msno
# import hvplot.pandas

PATH = './input/'
json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

nan_list = ["not available in demo dataset",
            "unknown.unknown",
            "(not provided)",
            "(not set)"
#             ,"Not Socially Engaged" # this last one is borderline 
           ]
nan_dict = {nl:np.nan for nl in nan_list}

# columns to drop : https://www.kaggle.com/c/google-analytics-customer-revenue-prediction/discussion/65691#387112
list_single_value = ['trafficSource.campaignCode', 'socialEngagementType', 'totals.visits']

def df_prep(file):
    df = pd.read_csv(file, dtype={'fullVisitorId': str, 'date': str}, 
            parse_dates=['date'],infer_datetime_format=True, nrows=None)
    
    for jc in json_cols:  # parse json  # Would probably be better with json_normalize from pandas
        flat_df = pd.DataFrame(df.pop(jc).apply(pd.io.json.loads).values.tolist())
        flat_df.columns = ['{}.{}'.format(jc, c) for c in flat_df.columns]
        df = df.join(flat_df)
    ad_df = df.pop('trafficSource.adwordsClickInfo').apply(pd.Series) # handle dict column
    ad_df.columns = ['adwords.{}'.format(c) for c in ad_df.columns]
    df = df.join(ad_df)
    df.replace(nan_dict, inplace=True) # handle disguised NaNs
    
    # Remove all-missing columns
    df.dropna(how="all",axis=1,inplace=True)
    
    df.drop([c for c in list_single_value if c in df.columns], axis=1, inplace=True)
    
# ### From : https://www.kaggle.com/mlisovyi/flatten-json-fields-smart-dump-data
    df['trafficSource.isTrueDirect'] = (df['trafficSource.isTrueDirect'].fillna(False)).astype(bool)
    df['totals.bounces'] = df['totals.bounces'].fillna(0).astype(np.uint8)
    df['totals.newVisits'] = df['totals.newVisits'].fillna(0).astype(np.uint8) # has NaNs ?
    df['totals.pageviews'] = df['totals.pageviews'].fillna(0).astype(np.uint16)
    
    # rename lat Long
    df.rename(columns={'geoNetwork.latitude':'Latitude', 'geoNetwork.longitude':"Longitude"},inplace=True)

    #parse unix epoch timestamp
    df.visitStartTime = pd.to_datetime(df.visitStartTime,unit='s',infer_datetime_format=True)
    
#     df.set_index(['fullVisitorId', 'sessionId'], inplace=True) # disabled for now

    df.drop(["sessionId"],axis=1,inplace=True)
    return df
%%time

train = df_prep(PATH+'train.csv')
print("train Shape: ",train.shape)
test = df_prep(PATH+'test.csv')
print("test Shape: ",test.shape)
display(train.head(7))
train.columns
train[['channelGrouping', 'date', 'fullVisitorId', 'visitId',
       'visitNumber', 'visitStartTime', 'device.browser',
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent', 'totals.bounces', 'totals.hits',
       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue',
        'trafficSource.adContent', 'trafficSource.campaign', 'trafficSource.isTrueDirect',
       'trafficSource.keyword', 'trafficSource.medium',
       'trafficSource.referralPath', 'trafficSource.source', 'adwords.page',
       'adwords.slot', 'adwords.gclId', 'adwords.adNetworkType']].nunique()
### Many variables only contain a single variable, remove them:
### change code version ; errors due to unhashable dicts
# columns = [col for col in train.columns if train[col].nunique() > 1] # can also be done with ".any() command"
# print(len(columns))
# train = train[columns]
# test = test[columns]
train.visitStartTime.describe()
#impute 0 for missing/NaNs of target column
train['totals.transactionRevenue'] = pd.to_numeric(train['totals.transactionRevenue'].fillna(0)) #.astype("float")
train['totals.transactionRevenue'].dtype
train.loc[train['totals.transactionRevenue']>0]['totals.transactionRevenue'].describe()
## https://www.kaggle.com/ashishpatel26/light-gbm-with-bayesian-style-parameter-tuning

gdf = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(9,7))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()
print("Train set: shape {} with {} unique users"
      .format(train.shape,train['fullVisitorId'].nunique()))
print("Test set: shape {} with {} unique users"
      .format(test.shape,test['fullVisitorId'].nunique()))
print("Users in both train and test set:",
      len(set(train.fullVisitorId.unique()).intersection(set(test.fullVisitorId.unique()))))
test.head()
print("orig test shape:",test.shape)
test_pred = test.set_index("visitStartTime",drop=False).groupby("fullVisitorId").last().reset_index().drop_duplicates("fullVisitorId")
print("pred ready test shape:",test_pred.shape)
test_pred.head()
# df2 = train.drop([#"date",
#                   "sessionId"
# #                   , "visitId" # ? 
#                  ],axis=1)

df2 = train.copy()

df2["sumLog_transactionRevenue"] = df2[["fullVisitorId","totals.transactionRevenue"]].groupby("fullVisitorId")["totals.transactionRevenue"].transform("sum")
# log transform target (we don't do log1P on purpose)! 
df2['sumLog_transactionRevenue'] = df2['sumLog_transactionRevenue'].apply(lambda x: np.log1p(x)) #.apply(lambda x: np.log(x) if x > 0 else x)
print("# unique visitor IDs : ", df2.fullVisitorId.nunique())
print("subset initial Data shape", df2.shape)
df2 = df2.set_index("visitStartTime",drop=False).groupby("fullVisitorId").last().drop("totals.transactionRevenue",axis=1).reset_index()
print("Data with target + only last entry in train per fullVisitorId:", df2.shape)
df2.tail()
df_context = pd.concat([train,test])
df_context.shape
# is enabling INDEXes, then keep index!
df2.to_csv("gstore_train_CLV_v1.csv.gz",index=False,compression="gzip")
# train.to_csv("gstore_train_v1.csv.gz",index=False,compression="gzip")
# test.to_csv("gstore_test_v1.csv.gz",index=False,compression="gzip")

df_context.to_csv("gstore_context_all_v1.csv.gz",index=False,compression="gzip")
test_pred.to_csv("gstore_test_Pred_v1.csv.gz",index=False,compression="gzip")
non_missing = len(train[~train['totals.transactionRevenue'].isnull()])
num_visitors = train[~train['totals.transactionRevenue'].isnull()]['fullVisitorId'].nunique()
print("totals.transactionRevenue has {} non-missing values or {:.3f}% (train set)"
      .format(non_missing, 100*non_missing/len(train)))
print("Only {} unique users have transactions or {:.3f}% (train set)"
      .format(num_visitors, num_visitors/train['fullVisitorId'].nunique()))
# Logn Distplot
revenue = train['totals.transactionRevenue'].dropna().astype('float64')
plt.figure(figsize=(10,4))
plt.title("Natural log Distribution - Transactions revenue")
ax1 = sns.distplot(np.log(revenue), color="#006633", fit=norm)
# Log10 Distplot
plt.figure(figsize=(10,4))
plt.title("Log10 Distribution - Transactions revenue")
ax1 = sns.distplot(np.log10(revenue), color="#006633", fit=norm)
target_df = pd.read_csv('../input/train.csv', usecols=['totals'])
flat_df = pd.io.json.json_normalize(target_df.totals.apply(json.loads))
flat_df['transactionRevenue'] = flat_df.transactionRevenue.astype(np.float32)
flat_df.transactionRevenue.isnull().sum()/flat_df.shape[0]
flat_df.fillna(0, inplace=True)
flat_dft.hist('transactionRevenue', bins=24) #.hvplo
flat_df.replace(0, np.NaN, inplace=True)
flat_df.hist('transactionRevenue', bins=25) #.hvplot