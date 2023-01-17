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
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import datetime as datetime
%%time 
train_data = load_df()
test_data = load_df("../input/test.csv")
train_data.head()
test_data.head()
train_data.describe()
list(train_data.columns.values)
(train_data['channelGrouping'].value_counts()).plot(kind='bar')
train_data['date'] = pd.to_datetime(train_data['date'], format="%Y%m%d")
(train_data['date'].value_counts()).plot()
(train_data['socialEngagementType'].value_counts()).plot(kind ='bar')
plt.xticks(rotation='horizontal')
(train_data['visitNumber'].value_counts()).plot()
train_data.groupby('device.browser')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')

train_data.groupby('trafficSource.source')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('trafficSource.medium')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')

train_data.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('geoNetwork.subContinent')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('geoNetwork.networkDomain')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')

train_data["totals.transactionRevenue"] = train_data["totals.transactionRevenue"].astype('float')
gdf = train_data.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
Nonzero_instance = pd.notnull(train_data["totals.transactionRevenue"]).sum()
Unique_customer = (gdf["totals.transactionRevenue"] > 0).sum()

print("There were", Nonzero_instance, "instances of activities that involved non-zero revenues.")
print("The ratio of activities that involved non-zero revenues to total activities was", '{:.2%}'.format(Nonzero_instance/train_data.shape[0]))
print()
print("There were", Unique_customer, "instances of activities that involved non-zero revenues from unique customers.")
print("The ratio of unique activities that involved non-zero revenues to unique total activities was", '{:.2%}'.format(Unique_customer/gdf.shape[0]))
print("In the train set, there were", train_data.fullVisitorId.nunique(), "unique visitors. There were", train_data.shape[0], "total non-unique visitors.")
print("In the test set, there were", test_data.fullVisitorId.nunique(), "unique visitors. There were", test_data.shape[0], "total non-unique visitors.")
print("There were", len(set(train_data.fullVisitorId.unique()).intersection(set(test_data.fullVisitorId.unique()))), "common visitors in the two data sets.")

train_x = train_data.drop(['date', 'fullVisitorId', 'sessionId', 'visitId', 'totals.transactionRevenue', 'socialEngagementType', 'device.browserSize', 'device.browserVersion', 'device.deviceCategory',
        'device.flashVersion', 'device.isMobile','device.screenResolution',  'device.language',
        'device.mobileDeviceBranding', 'device.mobileDeviceInfo',
        'device.mobileDeviceMarketingName', 'device.mobileDeviceModel',
        'device.mobileInputSelector', 'device.operatingSystem',
        'device.operatingSystemVersion', 'device.screenColors','geoNetwork.cityId',
        'geoNetwork.continent', 'geoNetwork.latitude',
        'geoNetwork.longitude', 'geoNetwork.networkDomain',
        'geoNetwork.networkLocation', 'geoNetwork.region',
        'geoNetwork.subContinent', 'visitStartTime', 'totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.visits',
        'trafficSource.adContent',
        'trafficSource.adwordsClickInfo.adNetworkType',
        'trafficSource.adwordsClickInfo.criteriaParameters',
        'trafficSource.adwordsClickInfo.gclId',
        'trafficSource.adwordsClickInfo.isVideoAd',
        'trafficSource.adwordsClickInfo.page',
        'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
        'trafficSource.campaignCode', 'trafficSource.isTrueDirect',
        'trafficSource.keyword', 'trafficSource.referralPath'], axis=1)
train_y = train_data['totals.transactionRevenue']
train_x['totals.pageviews'] = train_x['totals.pageviews'].astype(float)
categorical_col = train_x.select_dtypes(include = [np.object]).columns
numerical_col = train_x.select_dtypes(include = [np.number]).columns
categorical_col, numerical_col
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_col:
    train_col = list(train_x[col].values.astype(str))
    le.fit(train_col)
    train_x[col] = le.transform(train_col)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for col in numerical_col :
    train_x[col] = (train_x[col] - np.mean(train_x[col])/(np.max(train_x[col]) - np.min(train_x[col])))
train_x.head()
from sklearn.model_selection import train_test_split
trainX, crossX, trainY, crossY = train_test_split(train_x.values, train_y, test_size = 0.25, random_state = 20)
trainX
import lightgbm as lgb

lgb_params = {"objective":"regression", "metric": "rmse", "num_leaves": 500, "learning_rate" : 0.02, "max_bin":500, "bagging_fraction" : 0.75, "feature_fraction" : 0.8, "bagging_frequency" : 9, "num_interation": 1200}
lgb_train = lgb.Dataset(trainX, label=trainY)
lgb_val = lgb.Dataset(crossX, label=crossY)
model = lgb.train(lgb_params, lgb_train, valid_sets=[lgb_val], early_stopping_rounds=100, verbose_eval=20)