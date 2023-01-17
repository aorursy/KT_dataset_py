# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime

from sklearn.model_selection import GridSearchCV

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
from os.path import join as pjoin

data_root = '../input/make-data-ready'
print(os.listdir(data_root))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb

from pprint import pprint
import math

from scipy.stats import kurtosis, skew

from IPython import embed
from IPython.terminal.embed import InteractiveShellEmbed

from sklearn.model_selection import KFold

import random
import seaborn as sns
import matplotlib.pyplot as plt
import shap
plt.rcParams['figure.figsize'] = (12,6)
def load_data(data='train',n=2):
    df = pd.DataFrame()
    for i in range(n) :
        if data=='train':
            if i > 8 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'train_{i}.pkl'))
        elif data=='test':
            if i > 2 :
                break
            dfpart = pd.read_pickle(pjoin(data_root,f'test_{i}.pkl'))
        df = pd.concat([df,dfpart])
        del dfpart
    return df
        
%%time
df_train = load_data(n=9)
df_test = load_data('test',n=4)
# delete date-related features
del_cols=[item for item in df_train.columns if "Date" in item]
df_train=df_train.drop(del_cols,axis=1).copy()
df_test=df_test.drop(del_cols,axis=1).copy()

df_all = pd.concat([df_train,df_test]).reset_index(drop=True)
print(f'# of columns has na value: {(df_test.isnull().sum().sort_values(ascending=False) > 0).sum()}')
# train_v2.csv - from August 1st 2016 to April 30th 2018.
# test_v2.csv - from May 1st 2018 to October 15th 2018.
# sample_submission_v2.csv - from December 1st 2018 to January 31st 2019

# divide df_all into
# df_train_x - from August 1st 2016 to June 30st 2018, 5+12+6=23 months
# <-> 1.5 month (Jul, Aug * 0.5)
# df_train_y -  from August 15th 2018 to October 15th 2018, 2 months
# -> divide new_test into in-sample and out-of-sample to avoid overfitting
#   (only includes visitors Jan 15st 2018- June 30th 2018)

df_train_x = df_all[df_all.date <= "2018-06-30"].copy()
df_train_y = df_all[df_all.date >= "2018-08-15"].copy()

id_train = df_all[(df_all.date >= "2018-01-15") & (df_all.date <= "2018-06-30")].fullVisitorId.drop_duplicates()
df_train_x = df_train_x[df_train_x.fullVisitorId.isin(id_train)].copy()
df_train_y = df_train_y[df_train_y.fullVisitorId.isin(id_train)].copy()

# Apply the trained model to 
# df_test_x - from November 15th 2016 to October 15th 2018, 23 months
# <-> 1.5 month (Oct *0.5, Nov)
# df_test_y - from December 1st 2018 to January 31st 2019, 2 months 
#   (only includes visitors May 1st 2018- Oct 15th 2018)

df_test_x = df_all[(df_all.date >= "2016-11-15") & (df_all.date <= "2018-10-15")].copy()
id_test = df_all[(df_all.date >= "2018-05-01") & (df_all.date <= "2018-10-15")].fullVisitorId.drop_duplicates()
df_test_x = df_test_x[df_test_x.fullVisitorId.isin(id_test)].copy()
print({"all":df_all.shape,
       "train_x":df_train_x.shape,
       "train_y":df_train_y.shape,
       "test_x":df_test_x.shape,
       "id_train":id_train.shape,
       "id_test":id_test.shape})
#sns.lineplot(data=df_train_y.groupby("date")["fullVisitorId"].count())

#df_all.groupby("fullVisitorId")["totals_transactionRevenue"].sum().describe()
#df_train.groupby("fullVisitorId")["totals_transactionRevenue"].sum().describe()
#df_test.groupby("fullVisitorId")["totals_transactionRevenue"].sum().describe()
#df_all.groupby("fullVisitorId")["device_deviceCategory"].agg(lambda x:x.value_counts().index[0])

df_all.head()
%time
global df_train_x_encoded,df_train_y_encoded, df_all_encoded, df_test_x_encoded
df_train_x_encoded = df_train_x.copy()
df_train_y_encoded = df_train_y.copy()
df_all_encoded = df_all.copy()
df_test_x_encoded = df_test_x.copy()

## agg for last

# date
# the end of day of the dataset 10-15
import datetime
df_train_x_encoded["date"]=(datetime.datetime(2018,6,30)-df_train_x_encoded.date).dt.days.astype("int64")/365
df_train_y_encoded["date"]=(datetime.datetime(2018,10,15)-df_train_y_encoded.date).dt.days.astype("int64")/365
df_test_x_encoded["date"]=(datetime.datetime(2018,10,15)-df_test_x_encoded.date).dt.days.astype("int64")/365

# last_cols =  ["channelGrouping", "device_browser", 
#             "device_deviceCategory", "device_operatingSystem", 
#             "geoNetwork_city", "geoNetwork_continent", 
#             "geoNetwork_country", "geoNetwork_metro",
#             "geoNetwork_networkDomain", "geoNetwork_region", 
#             "geoNetwork_subContinent", "trafficSource_adContent", 
#             "trafficSource_adwordsClickInfo.adNetworkType", 
#             "trafficSource_adwordsClickInfo.gclId", 
#             "trafficSource_adwordsClickInfo.slot", "trafficSource_campaign",
#             "trafficSource_keyword", "trafficSource_medium", 
#             "trafficSource_referralPath", "trafficSource_source",
#             'trafficSource_adwordsClickInfo.isVideoAd',
#             'trafficSource_isTrueDirect', 'device_isMobile'] + ["fullVisitorId","date"]

cat_cols=["channelGrouping","device_browser","device_deviceCategory",
              "device_operatingSystem","geoNetwork_city","geoNetwork_continent","geoNetwork_country","trafficSource_medium",
              "trafficSource_adwordsClickInfo.isVideoAd","trafficSource_isTrueDirect","device_isMobile"]
last_cols = cat_cols + ["fullVisitorId","date"]

df_train_x_agg_last = df_train_x_encoded[last_cols].groupby("fullVisitorId",as_index=False).last().sort_values("fullVisitorId").reset_index(drop=True).copy()
df_test_x_agg_last = df_test_x_encoded[last_cols].groupby("fullVisitorId",as_index=False).last().sort_values("fullVisitorId").reset_index(drop=True).copy()
df_train_y_agg_last = df_train_y_encoded[last_cols].groupby("fullVisitorId",as_index=False).last().sort_values("fullVisitorId").reset_index(drop=True).copy()
import category_encoders as ce

ec= ce.OrdinalEncoder(cols=cat_cols,handle_unknown='impute')

ec.fit(pd.concat([df_train_x_agg_last,df_train_y_agg_last,df_test_x_agg_last]))
df_train_x_agg_last=ec.transform(df_train_x_agg_last)
df_train_y_agg_last=ec.transform(df_train_y_agg_last)
df_test_x_agg_last=ec.transform(df_test_x_agg_last)
df_train_x
%%time

num_cols = [item for item in df_train.columns if "totals" in item]
sum_cols = num_cols + ["fullVisitorId"]

df_train_x_agg_sum = df_train_x_encoded[sum_cols].groupby("fullVisitorId",as_index=False).sum().sort_values("fullVisitorId").reset_index(drop=True).copy()
df_test_x_agg_sum = df_test_x_encoded[sum_cols].groupby("fullVisitorId",as_index=False).sum().sort_values("fullVisitorId").reset_index(drop=True).copy()
df_train_y_agg_sum = df_train_y_encoded[sum_cols].groupby("fullVisitorId",as_index=False).sum().sort_values("fullVisitorId").reset_index(drop=True).copy()

# totals_transactionRevenue
df_train_x_agg_sum['totals_transactionRevenue'] = np.log1p(df_train_x_agg_sum['totals_transactionRevenue'])
df_test_x_agg_sum['totals_transactionRevenue'] = np.log1p(df_test_x_agg_sum['totals_transactionRevenue'])
df_train_y_agg_sum['totals_transactionRevenue'] = np.log1p(df_train_y_agg_sum['totals_transactionRevenue'])

df_train_x_agg = pd.merge(df_train_x_agg_sum,df_train_x_agg_last, how='left',on="fullVisitorId").sort_values("fullVisitorId").reset_index(drop=True).copy()
df_train_y_agg = pd.merge(df_train_y_agg_sum,df_train_y_agg_last, how='left',on="fullVisitorId").sort_values("fullVisitorId").reset_index(drop=True).copy()
df_test_x_agg = pd.merge(df_test_x_agg_sum,df_test_x_agg_last, how='left',on="fullVisitorId").sort_values("fullVisitorId").reset_index(drop=True).copy()

df_train_y_agg=pd.merge(id_train.to_frame(), df_train_y_agg, how='left',on="fullVisitorId").sort_values("fullVisitorId").reset_index(drop=True).copy()
df_train_y_agg["totals_transactionRevenue"]=df_train_y_agg["totals_transactionRevenue"].fillna(0)
print({"all":df_all.shape,
       "train_x":df_train_x_agg.shape,
       "train_y":df_train_y_agg.shape,
       "test_x":df_test_x_agg.shape,
       "id_train":id_train.shape,
       "id_test":id_test.shape})
df_train_x_agg[df_train_x_agg.totals_transactionRevenue>0]
# data setting

train_x = df_train_x_agg.drop(["fullVisitorId"],axis=1).values#[0:10000]
train_y = df_train_y_agg.totals_transactionRevenue.values#[0:10000]

test_x_id = df_test_x_agg.fullVisitorId.values
test_x = df_test_x_agg.drop(["fullVisitorId"],axis=1).values

# Grid Search
grid_params={'max_depth': [4,5,6],
        'subsample': [0.95],
        'colsample_bytree': [1.0]
}

xgb_model = xgb.XGBClassifier(early_stopping_rounds=50,eval_metric="rmse",verbose=3,n_jobs=-1)

grid_search = GridSearchCV(xgb_model,
                  grid_params,
                  cv=3,
                  scoring="neg_mean_squared_error",
                  n_jobs=1,
                  verbose=3)
grid_search.fit(train_x,train_y)
predict = grid_search.predict(test_x)
predict_df=pd.DataFrame()
predict_df["fullVisitorId"]=test_x_id
predict_df["PredictedLogRevenue"]=predict
predict_df.to_csv("xgboost_cv.csv", index=False)