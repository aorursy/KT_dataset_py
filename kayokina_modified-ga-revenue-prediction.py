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
"""Project Overview: 
For my Capstone Project I’m using dataset available for “Google Analytics Customer Revenue Prediction” competition on Kaggle.com but with a different task (as kaggle has changed the task recently). 
My goal is to predict spend of GStore customers in test data set.

Problem Statement: 
In this work, we are predicting the natural log of the transactions in test set:  PredictedLogRevenue.
"""
#IMPORTING REQUIRED LIBRARIES
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
import math
import random # random is to generate random values
from plotly.offline import init_notebook_mode, iplot, plot 
import plotly.graph_objs as go 
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer 
from sklearn.model_selection import RandomizedSearchCV

import lightgbm as lgb 

import gc
gc.enable()

import warnings
warnings.filterwarnings("ignore")
%matplotlib inline
""" 1. FLATTEN JSON """
# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields

def load_df(csv_path='../input/train.csv', JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource'], load_all=False):
    if not load_all:
        df = pd.read_csv(csv_path, 
                         converters={column: json.loads for column in JSON_COLUMNS}, 
                         dtype={'fullVisitorId': np.str}, nrows=250000)
    else:
        df = pd.read_csv(csv_path, 
                         converters={column: json.loads for column in JSON_COLUMNS}, 
                         dtype={'fullVisitorId': np.str})
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df


%%time
train = load_df("../input/train_v2.csv")
train.to_csv('train_new_flat.gz', compression='gzip',index=False)

%%time
test = load_df("../input/test_v2.csv", load_all=True)
test.to_csv('test_new_flat.gz', compression='gzip',index=False)
"""2. LOAD PREPARED DATA"""

# files 
train_file = 'train_new_flat.gz'
test_file = 'test_new_flat.gz'
#sample_submission_file = 'sample_submission_v2.csv'

# load and view files:
train_df = pd.read_csv(train_file, compression='gzip', dtype={'fullVisitorId': np.str})
train_df.head(1).T
train_df.shape
train_df.fullVisitorId.nunique() #220'374 out of 250K

# load and view files:
test_df = pd.read_csv(test_file, compression='gzip', dtype={'fullVisitorId': np.str})
test_df.head(1).T
test_df.shape
test_df.fullVisitorId.nunique() #296530 out of 401589

# difference in columns:
train_df.columns.difference(test_df.columns)
#FUNCTION FOR CALCULATING RSME score:
def rsme(y,pred):
    return(mean_squared_error(y,pred)**0.5)
"""GET TARGET VARIABLE  totals.transactionRevenue (and not the totals.total...)"""
len_train = len(train_df)

#COMBINING TRAIN AND TEST DATASET
df_combi=pd.concat([train_df,test_df],ignore_index=True)

# Store target variable of training data in a safe place
total_revenue = df_combi['totals.transactionRevenue'].fillna(0).astype(float)
total_revenue.describe()

# logarithm of target variable: transactionRevenue
target = total_revenue.apply(lambda x: np.log1p(x))
target.describe()

# split back into 2:
target_train = target.iloc[:len_train]
target_test = target.iloc[len_train:]

# check nulls:
df_combi.isnull().sum()
# Delete columns that are not flatten.. 
df_combi= df_combi.drop(['customDimensions', 'hits','trafficSource.campaignCode', 'totals.transactions', 'totals.totalTransactionRevenue', 'totals.transactionRevenue'], axis=1)

# convert to_datetime:
df_combi['date']=pd.to_datetime(df_combi.date, format='%Y%m%d')
df_combi['visitStartTime']=pd.to_datetime(df_combi.visitStartTime, unit='s')
df_combi[['date','visitStartTime']].head()

# convert to category:
df_combi['channelGrouping'] = df_combi['channelGrouping'].astype('category')
df_combi['device.deviceCategory'] = df_combi['device.deviceCategory'].astype('category')
df_combi['geoNetwork.continent'] = df_combi['geoNetwork.continent'].astype('category')   

# NULL treatment in numeric fileds:
df_combi['totals.pageviews'] = df_combi['totals.pageviews'].fillna(0).astype(int)
df_combi['totals.sessionQualityDim'] = df_combi['totals.sessionQualityDim'].fillna(0).astype(int)
df_combi['totals.timeOnSite'] = df_combi['totals.timeOnSite'].fillna(0).astype(int)
df_combi['trafficSource.adwordsClickInfo.page'] = df_combi['trafficSource.adwordsClickInfo.page'].fillna(0).astype(int)
df_combi['totals.bounces'] = df_combi['totals.bounces'].fillna('0').astype(int)
df_combi['totals.newVisits'] = df_combi['totals.newVisits'].fillna('0').astype(int)

# null in objects
df_combi['trafficSource.adwordsClickInfo.adNetworkType'] = df_combi['trafficSource.adwordsClickInfo.adNetworkType'].fillna('missing')
df_combi['trafficSource.adwordsClickInfo.slot'] = df_combi['trafficSource.adwordsClickInfo.slot'].fillna('missing')
df_combi['trafficSource.adContent'] = df_combi['trafficSource.adContent'].fillna('else')
df_combi['trafficSource.adwordsClickInfo.gclId'] = df_combi['trafficSource.adwordsClickInfo.gclId'].fillna('else')
df_combi['trafficSource.keyword'] = df_combi['trafficSource.keyword'].fillna('else')
df_combi['trafficSource.referralPath'] = df_combi['trafficSource.referralPath'].fillna('(not set)')
df_combi['trafficSource.adwordsClickInfo.isVideoAd'] = df_combi['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True)
df_combi['trafficSource.isTrueDirect'] = df_combi['trafficSource.isTrueDirect'].fillna(False)

# check if has only 1 value:
columns_to_remove = [col for col in df_combi.columns if df_combi[col].nunique() == 1]
print("Nb. of variables with unique value: {}".format(len(columns_to_remove)))

## check values:   
for col in columns_to_remove:
    print(col, df_combi[col].dtypes, df_combi[col].unique())
# Feature engineering
# new feature to help in EDA "has revenue>0 or not"
df_combi['is_non_zero_revenue'] = total_revenue>0 

# add time features:
df_combi['date_dayofweek'] =df_combi['date'].dt.dayofweek
df_combi['date_week'] = df_combi['date'].dt.week
df_combi['date_year'] =df_combi['date'].dt.year
df_combi['date_month'] =df_combi['date'].dt.month
# remove the rest of useless fields:
columns_to_remove = [col for col in df_combi.columns if df_combi[col].nunique() == 1]
df_combi= df_combi.drop(columns_to_remove, axis=1)
df_combi.head(1).T
df_combi.shape #(651589, 35)
df_combi.info()
#df_combi_clean.nunique()
df_combi.head()
df_combi.describe()

# show nulls
df_combi.isnull().sum()


# extract to csv
df_combi.to_csv("df_combi_clean.csv", index=False)
'III. EDA & feature engineering'

#How many have non-zero-revenue?   - 2687 out of 250000 = 1,07%
train_df['totals.transactionRevenue'][train_df['totals.transactionRevenue']>0].count()#7281
print("Number of non-zero transactions:")
print(train_df['totals.transactionRevenue'][train_df['totals.transactionRevenue']>0].count() ,"out of ",len_train, "or", train_df['totals.transactionRevenue'][train_df['totals.transactionRevenue']>0].count()/len_train*100, "%")

# check dates:
print("Min date in train set",pd.to_datetime(train_df.date.min(), format='%Y%m%d')) #20160805
# 2016-08-05
print("MAX date in train set",pd.to_datetime(train_df.date.max(), format='%Y%m%d')) 
#2018-04-29

# check dates:
test_df['date'].min() #20180501
test_df['date'].max() #20181015

print("Min date in test set",pd.to_datetime(test_df.date.min(), format='%Y%m%d')) #20160805
# 2018-05-01
print("MAX date in test set",pd.to_datetime(test_df.date.max(), format='%Y%m%d')) 
#2018-10-15 

# DF for EDA
timeseries_df = df_combi.iloc[:len_train]
timeseries_df['totals.transactionRevenue'] = total_revenue.iloc[:len_train]
timeseries_df.set_index('date', inplace=True)

# Plot the time series in your DataFrame
ax = timeseries_df['totals.transactionRevenue'].plot(color='blue')
ax.set_xlabel('Date')
ax.set_ylabel('transactionRevenue')
plt.title('transactionRevenue by date')
plt.show()

#Identifying Trends in Time Series
timeseries_df[['totals.transactionRevenue']].rolling(8).mean().plot()
plt.xlabel('Year', fontsize=20);

# certain peaks are happening according to month/week:
timeseries_df[['totals.transactionRevenue']].diff().plot()
plt.title('transactionRevenue difference')
plt.xlabel('Year');

timeseries_df.groupby('visitNumber')['totals.transactionRevenue'].mean().plot()
""" ADDITIONAL QUESTIONS TO ANSWER """ 

#1 HOW MANY OF CUSTOMERS in Train set HAVE MORE THAN 1 VISIT? 99'650 subs (19.3%) and 173'149 rows
print("Number of unique customers with more than 1 visit in train set:",
      timeseries_df.loc[timeseries_df['visitNumber']>1]['fullVisitorId'].nunique()) #41354 subs 
print ("out of total customers:",timeseries_df['fullVisitorId'].nunique() ) #220374 total
print("that is", round(timeseries_df.loc[timeseries_df['visitNumber']>1]['fullVisitorId'].nunique()/timeseries_df['fullVisitorId'].nunique()*100), "%")

#2 HOW MANY OF TEST/SUBMISSION SUBS ARE RETURNING USERS? (FROM TEST)
# 3510 customer of test-df exist in train_df:
test_df.fullVisitorId.isin(train_df.fullVisitorId).astype(int).sum()

# 3 DOES RETURNING CUSTOMER HAS BETTER PROBABILITY FOR TRANSACTION?
 ## yes, 2.5% of returning users had a transaction while only 0.6% of 1st time visitors

timeseries_df['is_returning_user'] = timeseries_df['visitNumber']>1
timeseries_df['is_returning_user'].head()
print("DOES RETURNING visitors spend more? (totals.transactionRevenue by return status)")
timeseries_df.groupby('is_returning_user')['is_non_zero_revenue'].agg(['sum', 'count', 'mean','std'])

#so avg revenue is higher per returning visitor:
timeseries_df.groupby('is_returning_user')['totals.transactionRevenue'].sum().plot.pie(autopct='%.1f%%')
plt.title('total revenue by returning visitor status')
plt.show()

# only 7281 visits (1.1%) had a transaction
only_with_transactions = timeseries_df.loc[timeseries_df['totals.transactionRevenue']>0]

# only 6719 out of 515999 (1.3%) subs had a transaction - some more >1 time
only_with_transactions.fullVisitorId.nunique()
timeseries_df.fullVisitorId.nunique()

# returning users with transacations had 1.7-times higher avg revenue than 1st time visitors
only_with_transactions.groupby('is_returning_user')['totals.transactionRevenue'].mean().plot('bar', title='AVG Revenue per Transaction')
only_with_transactions.groupby('is_returning_user')['totals.transactionRevenue'].mean()
# 4. HOW GEOGRAPHY IMPACT THE REVENUE?
print("Description of SubContinent count: ")
print(train_df['geoNetwork.subContinent'].value_counts()[:8]) # printing the top 7 percentage of browsers

# seting the graph size
plt.figure(figsize=(16,7))

# let explore the browser used by users
sns.countplot(train_df[train_df['geoNetwork.subContinent']\
                       .isin(train_df['geoNetwork.subContinent']\
                             .value_counts()[:15].index.values)]['geoNetwork.subContinent'], palette="hls") # It's a module to count the category's
plt.title("TOP 15 most frequent SubContinents", fontsize=20) # seting the title size
plt.xlabel("subContinent Names", fontsize=18) # seting the x label size
plt.ylabel("SubContinent Count", fontsize=18) # seting the y label size
plt.xticks(rotation=45) # Adjust the xticks, rotating the labels

plt.show() #use plt.show to render the graph that we did above

# Show device count and revenue distribution : 
# Initialize Figure and Axes object
fig, ax = plt.subplots(figsize=(10,4))
plt.subplot(2,1,1)
_ = sns.barplot(x='device.deviceCategory', y ='totals.transactionRevenue',data=train_df)
plt.subplot(2,1,2)
_ = sns.countplot(x='device.deviceCategory', data=train_df )
_rev_per_channel.plot('bar')
plt.show()

# Show channelGrouping count and revenue distribution : 
# Initialize Figure and Axes object
fig, ax = plt.subplots(figsize=(10,4))
plt.subplot(2,1,1)
_ = sns.barplot(x='channelGrouping', y ='totals.transactionRevenue',data=train_df)
plt.subplot(2,1,2)
_ = sns.countplot(x='channelGrouping', data=train_df )
_rev_per_channel.plot('bar')
plt.show()
# AVG revenue per channel count - incorrect - take only train or non-zero revenue count
rev = train_df.groupby('channelGrouping')['totals.transactionRevenue'].sum()
cnt = train_df.groupby('channelGrouping')['totals.transactionRevenue'].count()
rev_per_channel = rev/cnt
rev_per_channel.plot('bar')
# continent :
fig, ax = plt.subplots(figsize=(8,8))
plt.subplot(2,1,1)
_ = sns.barplot(x='geoNetwork.continent', y ='totals.transactionRevenue',data=train_df)
plt.subplot(2,1,2)
_ = sns.countplot(x='geoNetwork.continent', data=train_df )

# ad network type:
fig, ax = plt.subplots(figsize=(8,8))
plt.subplot(2,1,1)
_ = sns.barplot(x='merged_df', y ='totals_transactionRevenue',data=train_df)
plt.subplot(2,1,2)
_ = sns.countplot(x='merged_df', data=train_df )

# by trafficSource.medium
fig, ax = plt.subplots(figsize=(8,8))
plt.subplot(2,1,1)
_ = sns.barplot(x='trafficSource.medium', y ='totals.transactionRevenue',data=train_df)
plt.subplot(2,1,2)
_ = sns.countplot(x='trafficSource.medium', data=train_df )
" V PREPARE DATA FOR MODEL"

# delete irrelevant columns: 'sessionId' was del before
excluded = ['date',
            'fullVisitorId', 
            'visitId', 
            'visitStartTime',
            'is_non_zero_revenue']


df_for_model = df_combi.drop(excluded, axis=1)

# check & treat nulls:
df_for_model.isnull().sum()
df_for_model.info()
categorical_features =  ['channelGrouping',
                         'device.browser',
                         'device.deviceCategory',
                         'device.isMobile',
                         'device.operatingSystem',
                         'geoNetwork.city',
                         'geoNetwork.continent',
                         'geoNetwork.country',
                         'geoNetwork.metro',
                         'geoNetwork.networkDomain',
                         'geoNetwork.region',
                         'geoNetwork.subContinent',
                         'trafficSource.adContent',
                         'trafficSource.adwordsClickInfo.adNetworkType',
                         'trafficSource.adwordsClickInfo.gclId',
                         'trafficSource.adwordsClickInfo.isVideoAd',
                         'trafficSource.adwordsClickInfo.slot',
                         'trafficSource.campaign',
                         'trafficSource.isTrueDirect',
                         'trafficSource.keyword',
                         'trafficSource.medium',
                         'trafficSource.referralPath',
                         'trafficSource.source']

df_for_model_cat = df_for_model.copy()
feature_names = df_for_model_cat.columns.tolist()
# Encode catrgorical variables:
le = LabelEncoder()

for col in categorical_features:
    le = LabelEncoder()
    df_for_model_cat[col] = le.fit_transform(df_for_model_cat[col].astype(str))

# split back into 2 DF
df_train_clean = df_for_model_cat.iloc[:len(train_df)]
df_train_clean.shape #(250000, 35)
df_test_clean = df_for_model_cat.iloc[len(train_df):]
df_test_clean.shape #(401589, 35)

#preparing the data for models:
x_train= df_train_clean.values
y_train = target_train.values

x_test = df_test_clean.values
y_test = target_test.values
""" splitting into Kfolds by time series"""

from sklearn.model_selection import TimeSeriesSplit 
tscv = TimeSeriesSplit(n_splits=3)
print(tscv)  


for split_train_x_index, split_val_x_index in tscv.split(x_train):
    print("split_train_x_index:", split_train_x_index, "split_val_x:", split_val_x_index)

split_train_x, split_val_x = x_train[split_train_x_index], x_train[split_val_x_index]
split_train_y, split_val_y = y_train[split_train_x_index], y_train[split_val_x_index]
""" DUMMY MODEL """

# dummy model
zeros_y = np.zeros(len(target_test))
 
from sklearn.metrics import mean_squared_error
from math import sqrt

rms_zeros_y = sqrt(mean_squared_error(target_test, zeros_y)) 
#  1.9055877487314425
""" GradientBoostingRegressor BENCHMARK MODEL """

from sklearn.ensemble import GradientBoostingRegressor
 
#Let's go instantiate, fit and predict. 
gbrt=GradientBoostingRegressor(n_estimators=100, random_state=42)
 
gbrt.fit(x_train, y_train) 

gbrt.train_score_ 
gbrt.loss_
gbrt.score(x_train, y_train) # 0.32880548481994776

feature_imp_gbrt = pd.DataFrame(sorted(zip(gbrt.feature_importances_,df_for_model_cat.columns)), columns=['Value','Feature'])

plt.figure(figsize=(20, 10))
sns.barplot(x="Value", y="Feature", data=feature_imp_gbrt.sort_values(by="Value", ascending=False))
plt.title('GradientBoostingRegressor Features')
plt.tight_layout()
plt.show()
plt.savefig('gbrt_importances-01.png')

# Evaluation of gbrt
y_pred_gbrt=gbrt.predict(x_test) 
rms_gbrt = sqrt(mean_squared_error(y_test, y_pred_gbrt))  #1.139185
""" LightGBM MODEL - basic """

lgb_train = lgb.Dataset(split_train_x, split_train_y, feature_name=feature_names, categorical_feature=categorical_features )
lgb_eval = lgb.Dataset(split_val_x, split_val_y, reference=lgb_train )

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 2
}


print('Start training...')

# train
lgbm_kfold = lgb.train(params,
                lgb_train,
                num_boost_round=500,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

lgb.plot_importance(lgbm_kfold, figsize=(15, 10))
plt.show()

print('Save model...') 
# save model to file
lgbm_kfold.save_model('lgbm_kfold.txt')

# save pickle model: 
import pickle
filename = 'lgbm_kfold.sav'
pickle.dump(lgbm_kfold, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
#gbm_loaded_model = pickle.load(open(filename, 'rb'))


# params of the model:
lgbm_kfold.num_feature() #35 
lgbm_kfold.num_trees() #58

lgb.plot_importance(lgbm, figsize=(15, 10))
plt.show()

# predict
y_pred_lgbm_kfold = lgbm_kfold.predict(x_test, num_iteration=lgbm_kfold.best_iteration)

# Evaluation of LGBM
rms_lgbm_kfold = rsme(y_test, y_pred_lgbm_kfold)  #1.6268
""" FINE_TUNING OF LightGBM for final model"""

param_grid = {
        'n_estimators': [100, 500],
         'num_leaves': [15, 31, 63, 127],
        'learning_rate':[0.01, 0.05, 0.1],
        'max_depth': [5, 10, 15, 20, 30, 35,-1],
        'min_data_in_leaf': [30, 50, 100, 300, 5000],
        'drop_rate' : [0.1, 0.2, 0.3],
        'lambda_l1': [0, 0.2, 1],
        'lambda_l2': [0, 0.2, 1],
        'feature_fraction': [0.7, 0.9],
        'bagging_fraction': [0.8, 0.9],
        'bagging_freq': [5, 10]}


lgbm_estimator = lgb.LGBMRegressor(boosting='gbdt' ,  random_state=42)

#create scoring for Gridsearch
rmse_scorer = make_scorer(rsme, greater_is_better=False)

r_search = RandomizedSearchCV(estimator=lgbm_estimator, 
                             param_distributions=param_grid, 
                             scoring=rmse_scorer,
                             cv = tscv.get_n_splits([x_train,y_train]),
                             random_state=42, verbose=2)

lgbm_search_model = r_search.fit(x_train,y_train)

# save pickle model: 
import pickle
filename = 'lgbm_search_model.sav'
pickle.dump(lgbm_search_model, open(filename, 'wb'))


print("BEST PARAMETERS: " + str(lgbm_search_model.best_params_))
print("BEST CV SCORE: " + str(lgbm_search_model.best_score_))


# Predict (after fitting RandomizedSearchCV is an estimator with best parameters)
y_pred_lgbm_search = r_search.predict(x_test)

# Evaluation of LGBM
rms_lgbm_kfold = rsme(y_test, y_pred_lgbm_search)  #1.6263

r_search.best_estimator_

""" SUBMIT"""
df_submit_final = test_df.loc[:,['fullVisitorId']]
df_submit_final['PredictedLogRevenue'] = y_pred_lgbm_search
df_submit_final['RealLogRevenue'] = y_test
df_submit_final.to_csv("df_submit_final.csv", index=False)


def plot_diff(X, diff):
    fig = plt.subplots(figsize=(18,6))
    plt.plot(diff, linewidth=1)
    plt.xticks(rotation=45)
    plt.show()

difference = np.subtract(y_test, y_pred_lgbm_search)

plot_diff(x_test, difference)
plt.xlabel('visit')
plt.ylabel('Difference btwn PredictedLogRevenue vs RealLogRevenue');