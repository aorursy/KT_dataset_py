import sys
print(sys.executable)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns

import os
import json
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.io.json import json_normalize
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation, Dropout
from keras import optimizers

%matplotlib inline
pd.options.display.max_columns = 999
def load_df(csv_path, nrows=None):
    USE_COLUMNS = [
        'channelGrouping', 'date', 'device', 'fullVisitorId', 'geoNetwork',
        'socialEngagementType', 'totals', 'trafficSource', 'visitId',
        'visitNumber', 'visitStartTime', 'customDimensions'
        #'hits'
    ]
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows, usecols=USE_COLUMNS)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
%%time
train = load_df("../input/train_v2.csv")
test = load_df("../input/test_v2.csv")


def add_time_features(df):
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')
    df['year'] = df['date'].apply(lambda x: x.year)
    df['month'] = df['date'].apply(lambda x: x.month)
    df['day'] = df['date'].apply(lambda x: x.day)
    df['weekday'] = df['date'].apply(lambda x: x.weekday())
    
    return df
train_df = train.copy()
test_df = test.copy()

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
# Convert target feature to 'float' type.
train['totals.transactionRevenue'] = train["totals.transactionRevenue"].astype('float')
train['totals.pageviews'] = train['totals.pageviews'].astype(float)
train['totals.hits'] = train['totals.hits'].astype(float)
test['totals.pageviews'] = test['totals.pageviews'].astype(float)
test['totals.transactionRevenue'] = test["totals.transactionRevenue"].astype('float')
test['totals.hits'] = test['totals.hits'].astype(float)
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

train["totals.transactionRevenue"].fillna(0, inplace=True)
train
# dropiing the unwanted columns
unwanted_columns = ['customDimensions', 'day']
train = train.drop(unwanted_columns, axis=1)
test = test.drop(unwanted_columns, axis=1)
constant_columns = [c for c in train.columns if train[c].nunique()<=1]
print('Columns with constant values: ', constant_columns)
train = train.drop(constant_columns, axis=1)
constant_columns = [c for c in test.columns if test[c].nunique()<=1]
print('Columns with constant values: ', constant_columns)
test = test.drop(constant_columns, axis=1)
train = train.drop('year', axis=1)
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
categorical_features = ['device.isMobile', 'month', 'weekday']
train = pd.get_dummies(train, columns=categorical_features)
test = pd.get_dummies(test, columns=categorical_features)
train, test = train.align(test, join='outer', axis=1)

# replace the nan values added by align for 0
train.replace(to_replace=np.nan, value=0, inplace=True)
test.replace(to_replace=np.nan, value=0, inplace=True)
#creating test and validation set
X_train = train[train['date']<=datetime.date(2017, 12, 31)]
X_val = train[train['date']>datetime.date(2017, 12, 31)]
# Get labels
Y_train = X_train['totals.transactionRevenue'].values
Y_val = X_val['totals.transactionRevenue'].values
X_train = X_train.drop(['totals.transactionRevenue'], axis=1)
X_val = X_val.drop(['totals.transactionRevenue'], axis=1)
test = test.drop(['totals.transactionRevenue'], axis=1)

a = ['channelGrouping', 'date', 'device.browser', 'device.deviceCategory',
       'device.operatingSystem', 'fullVisitorId', 'geoNetwork.city',
       'geoNetwork.continent', 'geoNetwork.country', 'geoNetwork.metro',
       'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent', 'totals.sessionQualityDim',
       'totals.timeOnSite', 'totals.totalTransactionRevenue',
       'totals.transactions', 'trafficSource.adContent',
       'trafficSource.adwordsClickInfo.adNetworkType',
       'trafficSource.adwordsClickInfo.gclId',
       'trafficSource.adwordsClickInfo.page',
       'trafficSource.adwordsClickInfo.slot', 'trafficSource.campaign',
       'trafficSource.keyword', 'trafficSource.medium',
       'trafficSource.referralPath', 'trafficSource.source', 'visitId','visitStartTime']
reduce_features = a
X_train = X_train.drop(reduce_features, axis=1)
X_val = X_val.drop(reduce_features, axis=1)
test = test.drop(reduce_features, axis=1)

X_train.head()
normalized_features = ['visitNumber', 'totals.hits', 'totals.pageviews', 
                       'mean_hits_per_day', 'mean_pageviews_per_day', 
                       'sum_hits_per_day', 'sum_pageviews_per_day']
# Normalize using Min-Max scaling
scaler = preprocessing.MinMaxScaler()
X_train[normalized_features] = scaler.fit_transform(X_train[normalized_features])
X_val[normalized_features] = scaler.transform(X_val[normalized_features])
test[normalized_features] = scaler.transform(test[normalized_features])
X_train.head()
print(X_train.dtypes)
Y_train = np.log1p(Y_train)
Y_val = np.log1p(Y_val)
Y_train.max()

# model random forest
from sklearn.metrics import mean_squared_error
from math import sqrt

x = X_train
y = Y_train

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(criterion='mse')
model.fit(x, y)
predictions = model.predict(X_val)

ms = mean_squared_error(Y_val, predictions)

print(ms)

from math import sqrt

rms = sqrt(ms)
print(rms)
# model for ada boost
from sklearn.ensemble import AdaBoostRegressor
model2 = AdaBoostRegressor(n_estimators=300)
model2.fit(X_train, Y_train)
predictions2 = model2.predict(X_val)
ms2 = mean_squared_error(Y_val, predictions2)
print(ms2)
print(sqrt(ms2))
# model for linear regression
from sklearn.linear_model import LinearRegression
model3 = LinearRegression()
model3.fit(X_train, Y_train)
predictions3 = model3.predict(X_val)
ms3 = mean_squared_error(Y_val, predictions3)
print(ms3)
from math import sqrt
print(sqrt(ms3))
# model for decsision trees
from sklearn.tree import DecisionTreeRegressor
model4 = DecisionTreeRegressor()
model4.fit(X_train, Y_train)
predictions4 = model4.predict(X_val)
ms4 = mean_squared_error(Y_val, predictions4)
print(ms4)
print(sqrt(ms4))




# MODEL DEEP LEARNING 
BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 0.0003
model = Sequential()
model.add(Dense(256, kernel_initializer='glorot_normal', activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(128, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(1))
adam = optimizers.adam(lr=LEARNING_RATE)
model.compile(loss='mse', optimizer=adam)
print('Dataset size: %s' % X_train.shape[0])
print('Epochs: %s' % EPOCHS)
print('Learning rate: %s' % LEARNING_RATE)
print('Batch size: %s' % BATCH_SIZE)
print('Input dimension: %s' % X_train.shape[1])
print('Features used: %s' % X_train.columns.values)
model.summary()
history = model.fit(x=X_train.values, y=Y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, 
                    verbose=1, validation_data=(X_val.values, Y_val))
val_predictions = model.predict(X_val)
mse = mean_squared_error(val_predictions, Y_val)
rmse = np.sqrt(mean_squared_error(val_predictions, Y_val))

print('Model validation metrics')
print('MSE: %.2f' % mse)
print('RMSE: %.2f' % rmse)


plot_metrics(history.history['loss'], history.history['val_loss'])
plot(history)
