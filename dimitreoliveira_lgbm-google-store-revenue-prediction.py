import os

import json

import datetime

import numpy as np

import pandas as pd

import lightgbm as lgb

import seaborn as sns

from ast import literal_eval

import matplotlib.pyplot as plt

from pandas.io.json import json_normalize

from sklearn.metrics import mean_squared_error



%matplotlib inline

pd.options.display.max_columns = 999
def add_time_features(df):

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')

    df['year'] = df['date'].apply(lambda x: x.year)

    df['month'] = df['date'].apply(lambda x: x.month)

    df['day'] = df['date'].apply(lambda x: x.day)

    df['weekday'] = df['date'].apply(lambda x: x.weekday())

    

    return df
def load_df(file_name = 'train_v2.csv', nrows = None):

    USE_COLUMNS = [

        'channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',

        'socialEngagementType', 'totals', 'trafficSource', 'visitId',

        'visitNumber', 'visitStartTime', 'customDimensions']



    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']

    df = pd.read_csv('../input/{}'.format(file_name),

                     converters={column: json.loads for column in JSON_COLUMNS}, 

                     dtype={'fullVisitorId': 'str'}, nrows=nrows, usecols=USE_COLUMNS)

    

    for column in JSON_COLUMNS:

        column_as_df = json_normalize(df[column])

        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]

        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

        

    # Normalize customDimensions

    df['customDimensions']=df['customDimensions'].apply(literal_eval)

    df['customDimensions']=df['customDimensions'].str[0]

    df['customDimensions']=df['customDimensions'].apply(lambda x: {'index':np.NaN,'value':np.NaN} if pd.isnull(x) else x)



    column_as_df = json_normalize(df['customDimensions'])

    column_as_df.columns = [f"customDimensions.{subcolumn}" for subcolumn in column_as_df.columns]

    df = df.drop('customDimensions', axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df
train = load_df("../input/train_v2.csv", nrows=1000000)

test = load_df("../input/test_v2.csv", nrows=1000000)
train.head().T
print('TRAIN SET')

print('Rows: %s' % train.shape[0])

print('Columns: %s' % train.shape[1])

print('Features: %s' % train.columns.values)

print()

print('TEST SET')

print('Rows: %s' % test.shape[0])

print('Columns: %s' % test.shape[1])

print('Features: %s' % test.columns.values)
train = add_time_features(train)

test = add_time_features(test)

# Convert feature types.

train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')

train['totals.hits'] = train['totals.hits'].astype(float)

test['totals.hits'] = test['totals.hits'].astype(float)

train['totals.pageviews'] = train['totals.pageviews'].astype(float)

test['totals.pageviews'] = test['totals.pageviews'].astype(float)
gp_fullVisitorId_train = train.groupby(['fullVisitorId']).agg('sum')

gp_fullVisitorId_train.head()
# Train

gp_fullVisitorId_train = train.groupby(['fullVisitorId']).agg('sum')

gp_fullVisitorId_train['fullVisitorId'] = gp_fullVisitorId_train.index

gp_fullVisitorId_train['mean_hits_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.hits'].transform('mean')

gp_fullVisitorId_train['mean_pageviews_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.pageviews'].transform('mean')

gp_fullVisitorId_train['sum_hits_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.hits'].transform('sum')

gp_fullVisitorId_train['sum_pageviews_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.pageviews'].transform('sum')

gp_fullVisitorId_train = gp_fullVisitorId_train[['fullVisitorId', 'mean_hits_per_day', 'mean_pageviews_per_day', 'sum_hits_per_day', 'sum_pageviews_per_day']]

train = train.join(gp_fullVisitorId_train, on='fullVisitorId', how='inner', rsuffix='_')

train.drop(['fullVisitorId_'], axis=1, inplace=True)



# Test

gp_fullVisitorId_test = test.groupby(['fullVisitorId']).agg('sum')

gp_fullVisitorId_test['fullVisitorId'] = gp_fullVisitorId_test.index

gp_fullVisitorId_test['mean_hits_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.hits'].transform('mean')

gp_fullVisitorId_test['mean_pageviews_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.pageviews'].transform('mean')

gp_fullVisitorId_test['sum_hits_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.hits'].transform('sum')

gp_fullVisitorId_test['sum_pageviews_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.pageviews'].transform('sum')

gp_fullVisitorId_test = gp_fullVisitorId_test[['fullVisitorId', 'mean_hits_per_day', 'mean_pageviews_per_day', 'sum_hits_per_day', 'sum_pageviews_per_day']]

test = test.join(gp_fullVisitorId_test, on='fullVisitorId', how='inner', rsuffix='_')

test.drop(['fullVisitorId_'], axis=1, inplace=True)
train.head()
time_agg = train.groupby('date')['totals.transactionRevenue'].agg(['count', 'sum'])

year_agg = train.groupby('year')['totals.transactionRevenue'].agg(['sum'])

month_agg = train.groupby('month')['totals.transactionRevenue'].agg(['sum'])

day_agg = train.groupby('day')['totals.transactionRevenue'].agg(['sum'])

weekday_agg = train.groupby('weekday')['totals.transactionRevenue'].agg(['count','sum'])
plt.figure(figsize=(20,7))

plt.ticklabel_format(axis='y', style='plain')

plt.ylabel('Sum transactionRevenue', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.scatter(time_agg.index.values, time_agg['sum'])

plt.show()
plt.figure(figsize=(20,7))

plt.ticklabel_format(axis='y', style='plain')

plt.ylabel('Frequency', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.scatter(time_agg.index.values, time_agg['count'])

plt.show()
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(20,7))

ax1.scatter(year_agg.index.values, year_agg['sum'])

ax1.locator_params(nbins=2)

ax1.ticklabel_format(axis='y', style='plain')

ax1.set_xlabel('Year', fontsize=12)



ax2.scatter(month_agg.index.values, month_agg['sum'])

ax2.locator_params(nbins=12)

ax2.ticklabel_format(axis='y', style='plain')

ax2.set_xlabel('Month', fontsize=12)



ax3.scatter(day_agg.index.values, day_agg['sum'])

ax3.locator_params(nbins=10)

ax3.ticklabel_format(axis='y', style='plain')

ax3.set_xlabel('Day', fontsize=12)



ax4.scatter(weekday_agg.index.values, weekday_agg['sum'])

ax4.locator_params(nbins=7)

ax4.ticklabel_format(axis='y', style='plain')

ax4.set_xlabel('Weekday', fontsize=12)



plt.tight_layout()

plt.show()
# Drop column that exists only in train data

train = train.drop(['trafficSource.campaignCode'], axis=1)

# Input missing transactionRevenue values

train["totals.transactionRevenue"].fillna(0, inplace=True)



test_ids = test["fullVisitorId"].values
# Unwanted columns

unwanted_columns = ['channelGrouping', 'customDimensions.index', 'customDimensions.value', 'fullVisitorId',

                   'visitId', 'visitNumber', 'visitStartTime',

                   'device.browser', 'device.browserSize', 'device.browserVersion',

                   'device.deviceCategory', 'device.flashVersion',

                   'device.language', 'device.mobileDeviceBranding',

                   'device.mobileDeviceInfo', 'device.mobileDeviceMarketingName',

                   'device.mobileDeviceModel', 'device.mobileInputSelector',

                   'device.operatingSystem', 'device.operatingSystemVersion',

                   'device.screenColors', 'device.screenResolution', 'geoNetwork.city',

                   'geoNetwork.cityId', 'geoNetwork.continent', 'geoNetwork.country',

                   'geoNetwork.latitude', 'geoNetwork.longitude', 'geoNetwork.metro',

                   'geoNetwork.networkDomain', 'geoNetwork.networkLocation',

                   'geoNetwork.region', 'geoNetwork.subContinent',       

                   'totals.sessionQualityDim', 'trafficSource.adContent',

                   'trafficSource.adwordsClickInfo.adNetworkType',

                   'trafficSource.adwordsClickInfo.criteriaParameters',

                   'trafficSource.adwordsClickInfo.gclId',

                   'trafficSource.adwordsClickInfo.page',

                   'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',

                   'trafficSource.isTrueDirect', 'trafficSource.keyword',

                   'trafficSource.medium', 'trafficSource.referralPath',

                   'trafficSource.source']



train = train.drop(unwanted_columns, axis=1)

test = test.drop(unwanted_columns, axis=1)

# Constant columns

constant_columns = [c for c in train.columns if train[c].nunique()<=1]

print('Columns with constant values: ', constant_columns)

train = train.drop(constant_columns, axis=1)

test = test.drop(constant_columns, axis=1)

# Columns with more than 50% null data

high_null_columns = [c for c in train.columns if train[c].count()<=len(train) * 0.5]

print('Columns more than 50% null values: ', high_null_columns)

train = train.drop(high_null_columns, axis=1)

test = test.drop(high_null_columns, axis=1)
print('TRAIN SET')

print('Rows: %s' % train.shape[0])

print('Columns: %s' % train.shape[1])

print('Features: %s' % train.columns.values)



print()

print('TEST SET')

print('Rows: %s' % test.shape[0])

print('Columns: %s' % test.shape[1])

print('Features: %s' % test.columns.values)
train.head()
categorical_features = ['device.isMobile','year', 'month', 'weekday', 'day']

train = pd.get_dummies(train, columns=categorical_features)

test = pd.get_dummies(test, columns=categorical_features)
# align both data sets (by outer join), to make they have the same amount of features,

# this is required because of the mismatched categorical values in train and test sets

train, test = train.align(test, join='outer', axis=1)



# replace the nan values added by align for 0

train.replace(to_replace=np.nan, value=0, inplace=True)

test.replace(to_replace=np.nan, value=0, inplace=True)
X_train = train[train['date']<=datetime.date(2017, 5, 31)]

X_val = train[train['date']>datetime.date(2017, 5, 31)]
# Get labels

Y_train = X_train['totals.transactionRevenue'].values

Y_val = X_val['totals.transactionRevenue'].values

X_train = X_train.drop(['totals.transactionRevenue'], axis=1)

X_val = X_val.drop(['totals.transactionRevenue'], axis=1)

test = test.drop(['totals.transactionRevenue'], axis=1)

# Log transform the labels

Y_train = np.log1p(Y_train)

Y_val = np.log1p(Y_val)
reduce_features = ['date']

X_train = X_train.drop(reduce_features, axis=1)

X_val = X_val.drop(reduce_features, axis=1)

test = test.drop(reduce_features, axis=1)
X_train = X_train.astype('float32')

X_val = X_val.astype('float32')

test = test.astype('float32')
X_train.head()
params = {

"objective" : "regression",

"metric" : "rmse", 

"num_leaves" : 500,

"min_child_samples" : 20,

"learning_rate" : 0.005,

"bagging_fraction" : 0.6,

"feature_fraction" : 0.7,

"bagging_frequency" : 1,

"bagging_seed" : 1,

"lambda_l1": 3,

'min_data_in_leaf': 70

}
lgb_train = lgb.Dataset(X_train, label=Y_train)

lgb_val = lgb.Dataset(X_val, label=Y_val)

model = lgb.train(params, lgb_train, 10000, valid_sets=[lgb_train, lgb_val], early_stopping_rounds=100, verbose_eval=100)
# Make prediction on validation data.

val_predictions = model.predict(X_val, num_iteration=model.best_iteration)

# Get min and max values of the predictions and labels.

min_val = max(max(val_predictions), max(Y_val))

max_val = min(min(val_predictions), min(Y_val))

# Create dataframe with validation predicitons and labels.

val_df = pd.DataFrame({"Label":Y_val})

val_df["Prediction"] = val_predictions

# Plot data

sns.set(style="darkgrid")

sns.jointplot(y="Label", x="Prediction", data=val_df, kind="reg", color="m", height=10)

plt.plot([min_val, max_val], [min_val, max_val], 'm--')

plt.show()
val_predictions[val_predictions<0] = 0

mse = mean_squared_error(val_predictions, Y_val)

rmse = np.sqrt(mean_squared_error(val_predictions, Y_val))



print('Model validation metrics')

print('MSE: %.2f' % mse)

print('RMSE: %.2f' % rmse)
lgb.plot_importance(model, figsize=(15, 10))

plt.show()
predictions = model.predict(test, num_iteration=model.best_iteration)



submission = pd.DataFrame({"fullVisitorId":test_ids})

predictions[predictions<0] = 0

submission["PredictedLogRevenue"] = predictions

submission = submission.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()

submission.columns = ["fullVisitorId", "PredictedLogRevenue"]

submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"]

submission.to_csv("submission.csv", index=False)

submission.head(10)