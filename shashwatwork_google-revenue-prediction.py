# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import json

from pandas.io.json import json_normalize





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

from plotly.offline import init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

import seaborn as sns

from collections import Counter

import warnings

import featuretools as ft

import pandas_profiling

warnings.filterwarnings('ignore')



from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor,ExtraTreeRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.svm import SVR

from xgboost import XGBRegressor

from sklearn.model_selection import KFold,cross_val_score

from sklearn.metrics import make_scorer,r2_score,mean_squared_error



import keras

from keras.models import Sequential

from keras.layers import Dense

from tensorflow.keras.callbacks import EarlyStopping

from keras.wrappers.scikit_learn import KerasRegressor

def load_df(csv_path='../input/ga-customer-revenue-prediction/train_v2.csv', nrows=100000):

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

%%time

train_df = load_df()

test_df = load_df("../input/ga-customer-revenue-prediction/test_v2.csv")
train_df.head()
test_df.head()
#Looking data format and types

print(train_df.info())



# printing test info()

print(test_df.info())
null_feat = pd.DataFrame(len(train_df['fullVisitorId']) - train_df.isnull().sum(), columns = ['Count'])



trace = go.Bar(x = null_feat.index, y = null_feat['Count'] ,opacity = 0.8, marker=dict(color = 'red',

        line=dict(color='#000000',width=1.5)))



layout = dict(title =  "Missing Values")

                    

fig = dict(data = [trace], layout=layout)

py.iplot(fig)
print(f"Total of Unique visitor is {train_df.fullVisitorId.nunique()}")
from datetime import datetime



# This function is to extract date features

def date_process(df):

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d") # seting the column as pandas datetime

    df["_weekday"] = df['date'].dt.weekday #extracting week day

    df["_day"] = df['date'].dt.day # extracting day

    df["_month"] = df['date'].dt.month # extracting day

    df["_year"] = df['date'].dt.year # extracting day

    df['_visitHour'] = (df['visitStartTime'].apply(lambda x: str(datetime.fromtimestamp(x).hour))).astype(int)

    

    return df #returning the df after the transformations
train_df['totals.transactionRevenue'] = train_df['totals.transactionRevenue'].astype('float')

train_df["totals.transactionRevenue"].fillna(0, inplace=True)

train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index().plot()
# Get data

data = train_df['channelGrouping'].value_counts().sort_index(ascending=False)



# Create trace

trace = go.Bar(x = data.index,

               text = ['{:.1f} %'.format(val) for val in (data.values / train_df.shape[0] * 100)],

               textposition = 'auto',

               textfont = dict(color = '#000000'),

               y = data.values,

               marker = dict(color = '#db0000'))

# Create layout

layout = dict(title = 'Distribution Of {} Channel Grouping'.format(train_df.shape[0]),

              xaxis = dict(title = 'Channel'),

              yaxis = dict(title = 'Count'))

# Create plot

fig = go.Figure(data=[trace], layout=layout)

iplot(fig)
def missing_values(data):

    total = data.isnull().sum().sort_values(ascending = False) 

    percent = (data.isnull().sum() / data.isnull().count() * 100 ).sort_values(ascending = False) 

    df = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

    print("Total columns at least one Values: ")

    print (df[~(df['Total'] == 0)]) 

    

    print("\n Total of Sales % of Total: ", round((train_df[train_df['totals.transactionRevenue'] != np.nan]['totals.transactionRevenue'].count() / len(train_df['totals.transactionRevenue']) * 100),4))

    

    return 



missing_values(train_df)
train_df.head()
# Unwanted columns

col_to_drop = ['channelGrouping',

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



train_df = train_df.drop(col_to_drop, axis=1)

test_df = test_df.drop(col_to_drop, axis=1)
# Constant columns

constant_columns = [c for c in train_df.columns if train_df[c].nunique()<=1]

print('Columns with constant values: ', constant_columns)

train_df = train_df.drop(constant_columns, axis=1)

test_df = test_df.drop(constant_columns, axis=1)
high_null_columns = [c for c in train_df.columns if train_df[c].count()<=len(train_df) * 0.5]

print('Columns more than 50% null values: ', high_null_columns)

train = train_df.drop(high_null_columns, axis=1)

test = test_df.drop(high_null_columns, axis=1)
def convert_to_time(df):

    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d', errors='ignore')

    df['year'] = df['date'].apply(lambda x: x.year)

    df['month'] = df['date'].apply(lambda x: x.month)

    df['day'] = df['date'].apply(lambda x: x.day)

    df['weekday'] = df['date'].apply(lambda x: x.weekday())

    

    return df
train = convert_to_time(train_df)

test = convert_to_time(test_df)

# Convert feature types.

train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')

train['totals.hits'] = train['totals.hits'].astype(float)

test['totals.hits'] = test['totals.hits'].astype(float)

train['totals.pageviews'] = train['totals.pageviews'].astype(float)

test['totals.pageviews'] = test['totals.pageviews'].astype(float)
train
gp_fullVisitorId_train = train.groupby(['fullVisitorId']).agg('sum')

gp_fullVisitorId_train.head()
gp_fullVisitorId_train = train.groupby(['fullVisitorId']).agg('sum')

gp_fullVisitorId_train['fullVisitorId'] = gp_fullVisitorId_train.index

gp_fullVisitorId_train['mean_hits_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.hits'].transform('median')

gp_fullVisitorId_train['mean_pageviews_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.pageviews'].transform('median')

gp_fullVisitorId_train['sum_hits_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.hits'].transform('count')

gp_fullVisitorId_train['sum_pageviews_per_day'] = gp_fullVisitorId_train.groupby(['day'])['totals.pageviews'].transform('count')

gp_fullVisitorId_train = gp_fullVisitorId_train[['fullVisitorId', 'mean_hits_per_day', 'mean_pageviews_per_day', 'sum_hits_per_day', 'sum_pageviews_per_day']]

train = train.join(gp_fullVisitorId_train, on='fullVisitorId', how='inner',rsuffix='_')

train.drop(['fullVisitorId_'], axis=1, inplace=True)
gp_fullVisitorId_test = test.groupby(['fullVisitorId']).agg('count')

gp_fullVisitorId_test['fullVisitorId'] = gp_fullVisitorId_test.index

gp_fullVisitorId_test['mean_hits_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.hits'].transform('median')

gp_fullVisitorId_test['mean_pageviews_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.pageviews'].transform('median')

gp_fullVisitorId_test['sum_hits_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.hits'].transform('count')

gp_fullVisitorId_test['sum_pageviews_per_day'] = gp_fullVisitorId_test.groupby(['day'])['totals.pageviews'].transform('count')

gp_fullVisitorId_test = gp_fullVisitorId_test[['fullVisitorId', 'mean_hits_per_day', 'mean_pageviews_per_day', 'sum_hits_per_day', 'sum_pageviews_per_day']]

test = test.join(gp_fullVisitorId_test, on='fullVisitorId', how='inner',rsuffix='_')

test.drop(['fullVisitorId_'], axis=1, inplace=True)
display(train.columns)
categorical_features = ['device.isMobile','year', 'month', 'weekday', 'day']

train = pd.get_dummies(train,columns=categorical_features)

test = pd.get_dummies(test,columns=categorical_features)
train
test_ids = test["fullVisitorId"].values



train, test = train.align(test, join='outer', axis=1)



# replace the nan values added by align for 0

train.replace(to_replace=np.nan, value=0, inplace=True)

test.replace(to_replace=np.nan, value=0, inplace=True)
train
reduce_features = ['customDimensions','date','hits']

X_train = train.drop(reduce_features, axis=1)

test = train.drop(reduce_features, axis=1)

Y_train = X_train['totals.transactionRevenue'].values

Y_test = test['totals.transactionRevenue'].values
%%time

clfs = []

seed = 3



clfs.append(("LinearRegression", 

             Pipeline([("Scaler", StandardScaler()),

                       ("LogReg", LinearRegression())])))



clfs.append(("XGB",

             Pipeline([("Scaler", StandardScaler()),

                       ("XGB", XGBRegressor())]))) 

clfs.append(("KNN", 

             Pipeline([("Scaler", StandardScaler()),

                       ("KNN", KNeighborsRegressor())]))) 



clfs.append(("DTR", 

             Pipeline([("Scaler", StandardScaler()),

                       ("DecisionTrees", DecisionTreeRegressor())]))) 



clfs.append(("RFRegressor", 

             Pipeline([("Scaler", StandardScaler()),

                       ("RandomForest", RandomForestRegressor())]))) 



clfs.append(("GBRegressor", 

             Pipeline([("Scaler", StandardScaler()),

                       ("GradientBoosting", GradientBoostingRegressor(max_features=15, 

                                                                       n_estimators=600))]))) 



clfs.append(("EXT Regressor",

             Pipeline([("Scaler", StandardScaler()),

                       ("ExtraTrees", ExtraTreeRegressor())])))



scoring = 'r2'

n_folds = 10

msgs = []

results, names  = [], [] 



for name, model  in clfs:

    kfold = KFold(n_splits=n_folds, random_state=seed)

    cv_results = cross_val_score(model, X_train, Y_train, 

                                 cv=kfold, scoring=scoring, n_jobs=-1)    

    names.append(name)

    results.append(cv_results)    

    msg = "%s: %f (+/- %f)" % (name, cv_results.mean(),  

                               cv_results.std())

    msgs.append(msg)

    print(msg)
# Define error measure for official scoring : RMSE

scorer = make_scorer(mean_squared_error, greater_is_better = False)



def rmse_cv_train(model):

    rmse= np.sqrt(-cross_val_score(model, X_train, Y_train, scoring = scorer, cv = 10))

    return(rmse)



def rmse_cv_test(model):

    rmse= np.sqrt(-cross_val_score(model, test, Y_test, scoring = scorer, cv = 10))

    return(rmse)
lr = RandomForestRegressor(n_estimators=100)

lr.fit(X_train, Y_train)



# Look at predictions on training and validation set

print("RMSE on Training set :", rmse_cv_train(lr).mean())

print("RMSE on Test set :", rmse_cv_test(lr).mean())

y_train_pred = lr.predict(X_train)

y_test_pred = lr.predict(test)



# Plot residuals

plt.scatter(y_train_pred, y_train_pred - Y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, y_test_pred - Y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Residuals")

plt.legend(loc = "upper left")

plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")

plt.show()



# Plot predictions

plt.scatter(y_train_pred, Y_train, c = "blue", marker = "s", label = "Training data")

plt.scatter(y_test_pred, Y_test, c = "lightgreen", marker = "s", label = "Validation data")

plt.title("Linear regression")

plt.xlabel("Predicted values")

plt.ylabel("Real values")

plt.legend(loc = "upper left")

plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")

plt.show()
def model():    

    model = Sequential()

    model.add(Dense(128,input_dim = 63,activation='relu',kernel_initializer='normal'))

    model.add(Dense(64,activation='tanh',kernel_initializer='normal'))

    model.add(Dense(1,activation = 'linear'))

    model.compile(loss = 'mse',optimizer='adam',metrics=['mse','mae'])

    return model
estimators = []

estimators.append(('standardize', StandardScaler()))

estimators.append(('keras', KerasRegressor(build_fn=model, epochs=10, batch_size=128, verbose=1)))

pipeline = Pipeline(estimators)

pipeline.fit(X_train,Y_train)
y_pred= pipeline.predict(test)

fig, ax = plt.subplots()

ax.scatter(Y_test, y_pred)

ax.plot([Y_test.min(), Y_test.max()], [Y_test.min(), Y_test.max()], 'k--', lw=4)

ax.set_xlabel('Measured')

ax.set_ylabel('Predicted')

plt.show()
predictions = lr.predict(test)



submission = pd.DataFrame({"fullVisitorId":test_ids})

y_pred[y_pred<0] = 0

submission["PredictedLogRevenue"] = predictions

submission = submission.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()

submission.columns = ["fullVisitorId", "PredictedLogRevenue"]

submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"]

submission.to_csv("submission.csv", index=False)

submission.head(10)