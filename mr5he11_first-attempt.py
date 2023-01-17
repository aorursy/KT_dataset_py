import os
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn import neighbors
from pandas.io.json import json_normalize
%matplotlib notebook
import matplotlib.pyplot as plot
def load_df(csv_path='../input/train_v2.csv', nrows=None):
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
ga_customers_train = load_df(nrows=200000)
ga_customers_test = load_df(csv_path='../input/test_v2.csv')
ga_customers_train.head(5)
ga_customers_train = ga_customers_train.drop(columns=['customDimensions', 'hits'])
ga_customers_test = ga_customers_test.drop(columns=['customDimensions', 'hits'])
const_cols = [c for c in ga_customers_train.columns if ga_customers_train[c].nunique(dropna=False)==1 ]
ga_customers_train = ga_customers_train.drop(columns=const_cols)
ga_customers_test = ga_customers_test.drop(columns=const_cols)
ga_customers_train['totals.transactionRevenue'] = ga_customers_train['totals.transactionRevenue'].astype('float')
total_transactions_per_user = ga_customers_train.groupby('fullVisitorId')['totals.transactionRevenue'].sum().reset_index()
plot.figure(figsize=(8,6))
plot.scatter(range(total_transactions_per_user.shape[0]), np.sort(np.log1p(total_transactions_per_user["totals.transactionRevenue"].values)))
plot.xlabel('index', fontsize=12)
plot.ylabel('Transaction Revenue', fontsize=12)
plot.show()
real_customers = pd.notnull(ga_customers_train["totals.transactionRevenue"]).sum()
print("Percentage of real customers (people who bought something):")
print(real_customers/ga_customers_train.size*100)
ga_customers_train.dtypes
mixed_dataset = ga_customers_train.append(ga_customers_test, sort=False)
mixed_dataset['channelGrouping'] = mixed_dataset['channelGrouping'].fillna('none').astype('category')
#we don't train our model on Ids features, so we can avoid to handle their types
mixed_dataset['device.browser'] = mixed_dataset['device.browser'].fillna('none').astype('category')
mixed_dataset['device.deviceCategory'] = mixed_dataset['device.deviceCategory'].fillna('none').astype('category')
mixed_dataset['device.operatingSystem'] = mixed_dataset['device.operatingSystem'].fillna('none').astype('category')
mixed_dataset['geoNetwork.city'] = mixed_dataset['geoNetwork.city'].fillna('none').astype('|S2048')
mixed_dataset['geoNetwork.country'] = mixed_dataset['geoNetwork.country'].fillna('none').astype('unicode')
mixed_dataset['geoNetwork.continent'] = mixed_dataset['geoNetwork.continent'].fillna('none').astype('|S2048')
mixed_dataset['geoNetwork.metro'] = mixed_dataset['geoNetwork.metro'].fillna('none').astype('|S2048')
mixed_dataset['geoNetwork.networkDomain'] = mixed_dataset['geoNetwork.networkDomain'].fillna('none').astype('|S2048')
mixed_dataset['geoNetwork.region'] = mixed_dataset['geoNetwork.region'].fillna('none').astype('|S2048')
mixed_dataset['geoNetwork.subContinent'] = mixed_dataset['geoNetwork.subContinent'].fillna('none').astype('|S2048')
mixed_dataset['totals.bounces'] = pd.to_numeric(mixed_dataset['totals.bounces'].fillna(0))
mixed_dataset['totals.hits'] = pd.to_numeric(mixed_dataset['totals.hits'].fillna(0))
mixed_dataset['totals.newVisits'] = pd.to_numeric(mixed_dataset['totals.newVisits'].fillna(0))
mixed_dataset['totals.pageviews'] = pd.to_numeric(mixed_dataset['totals.pageviews'].fillna(0))
mixed_dataset['totals.sessionQualityDim'] = pd.to_numeric(mixed_dataset['totals.sessionQualityDim'].fillna(0))
mixed_dataset['totals.timeOnSite'] = pd.to_numeric(mixed_dataset['totals.timeOnSite'].fillna(0))
mixed_dataset['totals.transactionRevenue'] = mixed_dataset['totals.transactionRevenue'].fillna(0)
mixed_dataset['totals.totalTransactionRevenue'] = pd.to_numeric(mixed_dataset['totals.totalTransactionRevenue'].fillna(0))
mixed_dataset['totals.transactions'] = pd.to_numeric(mixed_dataset['totals.transactions'].fillna(0))
mixed_dataset['trafficSource.adContent'] = mixed_dataset['totals.transactions'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.adwordsClickInfo.adNetworkType'] = mixed_dataset['trafficSource.adwordsClickInfo.adNetworkType'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.adwordsClickInfo.gclId'] = mixed_dataset['trafficSource.adwordsClickInfo.gclId'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.adwordsClickInfo.isVideoAd'] = mixed_dataset['trafficSource.adwordsClickInfo.isVideoAd'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.adwordsClickInfo.page'] = pd.to_numeric(mixed_dataset['trafficSource.adwordsClickInfo.page'].fillna(0))
mixed_dataset['trafficSource.adwordsClickInfo.slot'] = mixed_dataset['trafficSource.adwordsClickInfo.slot'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.campaign'] = mixed_dataset['trafficSource.campaign'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.campaignCode'] = mixed_dataset['trafficSource.campaignCode'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.isTrueDirect'] = mixed_dataset['trafficSource.isTrueDirect'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.keyword'] = mixed_dataset['trafficSource.keyword'].fillna('none').astype('unicode')
mixed_dataset['trafficSource.medium'] = mixed_dataset['trafficSource.medium'].fillna('none').astype('|S2048')
mixed_dataset['trafficSource.referralPath'] = mixed_dataset['trafficSource.referralPath'].fillna('none').astype('unicode')
mixed_dataset['trafficSource.source'] = mixed_dataset['trafficSource.source'].fillna('none').astype('|S2048')
mixed_dataset.dtypes
les = []
features = ['channelGrouping', 
            'device.browser',
            'device.deviceCategory',
            'device.operatingSystem', 
            'geoNetwork.city', 
            'geoNetwork.country',
            'geoNetwork.metro',
            'geoNetwork.networkDomain', 
            'geoNetwork.region',
            'geoNetwork.continent',
            'geoNetwork.subContinent',
            'trafficSource.adContent',
            'trafficSource.adwordsClickInfo.adNetworkType',
            'trafficSource.adwordsClickInfo.gclId',
            'trafficSource.adwordsClickInfo.isVideoAd',
            'trafficSource.adwordsClickInfo.slot', 
            'trafficSource.campaign',
            'trafficSource.campaignCode',
            'trafficSource.isTrueDirect', 
            'trafficSource.keyword',
            'trafficSource.medium', 
            'trafficSource.referralPath',
            'trafficSource.source']
for feature in features:
    le = preprocessing.LabelEncoder()
    try:
        le.fit(mixed_dataset[feature])
        mixed_dataset[feature] = le.transform(mixed_dataset[feature])
        les.append(le)
    except (SystemError, UnicodeEncodeError):
       print("Error: can't handle " + feature + " data type (maybe it is a unicode string).")
print(mixed_dataset.values[0,:])
X_train = mixed_dataset.drop(columns=['totals.transactionRevenue','fullVisitorId', 'visitId']).values[0:ga_customers_train.shape[0],:]
X_test = mixed_dataset.drop(columns=['totals.transactionRevenue','fullVisitorId', 'visitId']).values[ga_customers_train.shape[0]:mixed_dataset.shape[0],:]
y = mixed_dataset['totals.transactionRevenue'].values[0:ga_customers_train.shape[0]]
regressor = DecisionTreeRegressor()
regressor.fit(X_train, y)
tree_result = regressor.predict(X_test)
random_forest = RandomForestRegressor(n_estimators=100, bootstrap=True)
random_forest.fit(X_train, y)
ra_predictions = random_forest.predict(X_test)
submission = ga_customers_test[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = ra_predictions
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test["PredictedLogRevenue"] = np.log1p(grouped_test["PredictedLogRevenue"])
grouped_test.to_csv('submit.csv',index=False)