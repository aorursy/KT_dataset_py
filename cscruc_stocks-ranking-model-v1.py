!pip install tushare



import tushare as ts

import pandas as pd

import numpy as np

import lightgbm as lgb

import datetime

from datetime import timedelta

from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import OneHotEncoder

import sys, os

import matplotlib.pylab as plt
#####################################################

## get all stocks list

all_codes = ts.get_today_all()['code']

codes_df = pd.DataFrame(all_codes)

codes_df["prefix"] = codes_df["code"].str.slice(0,3)

codes_df.groupby(["prefix"])["code"].count().sort_values(ascending=False)



## only contains code prefix in 002,600,603,000,601

codes_df = codes_df[codes_df["prefix"].isin(("002","600","603","000","601"))]

codes_df = codes_df.drop_duplicates()

codes_df.groupby(["prefix"])["code"].count().sort_values(ascending=False)
####################################################

## get all history data

all_hist_df = pd.DataFrame()

cnt = 0

for curr_code in codes_df["code"]:

    #if cnt > 10: break

    tmp_df = ts.get_hist_data(curr_code)

    if not (tmp_df is None):

        try:

            tmp_df["code"] = curr_code

            tmp_df["prefix"] = tmp_df["code"].str.slice(0,3)

            all_hist_df = pd.concat([all_hist_df, tmp_df])

            cnt += 1

        except:

            continue



## add column date

all_hist_df["date"] = all_hist_df.index.values
######################################################

#generate basic features from http://tushare.org/fundamental.html

basic_data = ts.get_stock_basics()

basic_data.reset_index(inplace=True)



#time to market -> datediff between time to market and now 

basic_data = basic_data[basic_data["timeToMarket"] > 0] #del NA

basic_data["timeToMarket"] = pd.to_datetime(basic_data["timeToMarket"].apply(str), format = "%Y%m%d").apply(lambda x: x.strftime('%Y-%m-%d'))



basic_features = ["industry","area","pe","outstanding","totals","totalAssets","liquidAssets","fixedAssets","reserved",

                  "reservedPerShare","esp","bvps","pb","undp","perundp","rev","profit","gpr","npr","holders"]

basic_data = basic_data[basic_features + ["code","timeToMarket"]]

basic_data = basic_data.drop_duplicates()

all_hist_df = pd.merge(all_hist_df, basic_data, left_on = ["code"], right_on = ["code"], suffixes=('', '_basic'))

all_hist_df["timeToMarketInterval"] = (pd.to_datetime(all_hist_df["date"]) - pd.to_datetime(all_hist_df["timeToMarket"])).apply(lambda x: x.days)



print(all_hist_df["timeToMarketInterval"].describe())

all_hist_df = all_hist_df[all_hist_df["timeToMarketInterval"] > 30] # del new stocks (timeToMarketInterval <= 30)

all_hist_df = all_hist_df[(all_hist_df["open"] != all_hist_df["high"]) | (all_hist_df["open"] != all_hist_df["close"])] # del daily limit stocks
#########################################################

#feature combine & encode

features = ["open","close","high","low","ma5","ma10","ma20","v_ma5","v_ma10","v_ma20","prefix"] + basic_features

all_hist_df = all_hist_df[features + ["code","date"]]

categorical = ["prefix", "industry", "area"]



for feature in categorical:

    print(f'Transforming {feature}...')

    encoder = LabelEncoder()

    encoder.fit(all_hist_df[feature])

    all_hist_df[feature] = encoder.transform(all_hist_df[feature].astype(str))
#########################################################

# test set: stocks data on yesterday

yesterday = max(all_hist_df["date"])

all_yesterday_df = all_hist_df[all_hist_df["date"]==yesterday]



# label: price diff after 7 days

label = ["price_diff"]

seven_days = timedelta(days=7)

all_hist_df["date_7d"] = (pd.to_datetime(all_hist_df["date"]) + seven_days).apply(lambda x: x.strftime('%Y-%m-%d'))

all_hist_df = pd.merge(all_hist_df, all_hist_df[["code","close","date"]], left_on = ["code","date_7d"], 

                       right_on = ["code","date"], suffixes=('', '_groundtruth'))

all_hist_df.drop(["date_7d"], axis = 1, inplace=True)

all_hist_df = all_hist_df[all_hist_df["close_groundtruth"].isna()==False]

all_hist_df["price_diff"] = (all_hist_df["close_groundtruth"] - all_hist_df["high"]) / all_hist_df["high"]

all_hist_df = all_hist_df[all_hist_df["price_diff"].isna()==False]

all_hist_df[["price_diff"]].describe()

print(all_hist_df.shape)
###############################################################

#training params

params = {

    'objective' : 'regression',

    'metric' : 'rmse',

    'num_leaves' : 32,

    'max_depth': -1,

    'learning_rate' : 0.02,

    'feature_fraction' : 0.7,

    'verbosity' : -1

}

all_hist_df = all_hist_df.sample(frac=1) #shuffle

VALID_RATIO = 0.2

train = all_hist_df[:int(all_hist_df.shape[0] * (1 - VALID_RATIO))] #training set 

valid = all_hist_df[int(all_hist_df.shape[0] * (1 - VALID_RATIO)):] #valid set

print(train.shape)

print(valid.shape)

train.head()
################################################################

# lgb training

lgtrain = lgb.Dataset(train[features], train[label],feature_name=list(train[features].columns),categorical_feature = categorical)

lgtest = lgb.Dataset(valid[features], valid[label],feature_name=list(valid[features].columns),categorical_feature = categorical)

watchlist = [lgtrain,lgtest]

lgb_clf = lgb.train(params,lgtrain,num_boost_round=1000,verbose_eval=100,valid_sets=watchlist,early_stopping_rounds=50)





# top 100 history score

y_test_pred_log = lgb_clf.predict(valid[features])

valid["score"] = y_test_pred_log

valid = valid.sort_values("score",ascending=False)

valid[:100]
#################################################################

# feature importance

plt.figure(figsize=(12,6))

lgb.plot_importance(lgb_clf, max_num_features=30)

plt.title("Featurertances")

plt.show()
#################################################################

# predict data from yesterday

predict_value = lgb_clf.predict(all_yesterday_df[features])

all_yesterday_df["score"] = predict_value

all_yesterday_df = all_yesterday_df.sort_values("score",ascending=False)



# top 100 score 

all_yesterday_df[:100]
#guanglianda

all_yesterday_df[all_yesterday_df["code"]=="002410"]
#zijinkuangye

all_yesterday_df[all_yesterday_df["code"]=="601899"]