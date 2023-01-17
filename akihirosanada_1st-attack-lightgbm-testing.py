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
import matplotlib.pyplot as plt
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
market_train_df, news_train_df = env.get_training_data()
market_train_df.head()
market_train_df.shape
len(market_train_df.assetCode.unique())
news_train_df.head()
len(news_train_df.headline[2325409])
news_train_df.shape
def data_prep(market_df, news_df):
    market_df["date"] = market_df.time.dt.date
    market_df["returnsOpenPrevRaw1_to_volume"] = \
        market_df["returnsOpenPrevRaw1"] / market_df["volume"]
    market_df["close_to_open"] = market_df["close"] / market_df["open"]
    news_df["firstCreatedDate"] = news_df.firstCreated.dt.date
    news_df["firstAssetCode"] = news_df["assetCodes"].map(lambda x: list(eval(x))[0])
    news_df["headlineLen"] = news_df["headline"].apply(lambda x: len(x))
    lbl = {k: v for v,k in enumerate(news_df["headlineTag"].unique())}
    news_df["headlineTagT"] = news_df["headlineTag"].map(lbl)
    kcol = ["firstCreatedDate","firstAssetCode"]
    numcols = ["urgency","takeSequence","bodySize","companyCount","sentenceCount",
               "wordCount","firstMentionSentence","relevance","sentimentClass",
              "sentimentNegative","sentimentNeutral","sentimentPositive",
              "sentimentWordCount","noveltyCount12H","noveltyCount24H",
              "noveltyCount3D","noveltyCount5D","noveltyCount7D",
              "volumeCounts12H","volumeCounts24H","volumeCounts3D","volumeCounts5D",
              "volumeCounts5D","volumeCounts7D","headlineLen"]
    news_df = news_df.loc[:,news_df.columns.isin(numcols + kcol)]\
        .groupby(kcol, as_index = False).mean()
    market_df = pd.merge(market_df, news_df, how = "left", left_on = ["date","assetCode"],
                        right_on = ["firstCreatedDate","firstAssetCode"])
    lbl = {k:v for v,k in enumerate(market_df["assetCode"].unique())}
    market_df["assetCodeT"] = market_df["assetCode"].map(lbl)
    return market_df
#for save RAM
news_train_df = news_train_df.loc[news_train_df.time >= "2010-01-01 22:00:00+00:00"]
market_train_df = market_train_df.loc[market_train_df.time >= "2010-01-01 22:00:00+00:00"]
from time import time
t_start_prep = time()
market_train = data_prep(market_train_df, news_train_df)
print(market_train.shape)
t_end_prep = time()
print("time consumed for prep:", t_end_prep - t_start_prep)
market_train.columns
# feature variables
fcol = ['volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenPrevRaw1_to_volume', 'close_to_open', 'urgency',
       'takeSequence', 'bodySize', 'companyCount',
       'sentenceCount', 'wordCount', 'firstMentionSentence', 'relevance',
       'sentimentClass', 'sentimentNegative', 'sentimentNeutral',
       'sentimentPositive', 'sentimentWordCount', 'noveltyCount12H',
       'noveltyCount24H', 'noveltyCount3D', 'noveltyCount5D', 'noveltyCount7D',
       'volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D',
       'volumeCounts5D', 'volumeCounts7D', 'headlineLen']
market_train.head()
X = market_train[fcol].fillna(0).values
q1 = market_train.returnsOpenNextMktres10.quantile(0.25)
q2 = market_train.returnsOpenNextMktres10.quantile(0.5)
q3 = market_train.returnsOpenNextMktres10.quantile(0.75)
print("q3:",q3)
def classify(x):
    if x >= q3:
        return 3
    elif x >= q2:
        return 2
    elif x >= q1:
        return 1
    else:
        return 0
Y = market_train.returnsOpenNextMktres10.apply(classify)
Y.value_counts()
#clip
X[X > 1000000000] = 1000000000
X[X < -1000000000] = -1000000000
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y,test_size = 0.1,
                                                                   shuffle = False)
%%time
import lightgbm as lgb
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_test, Y_test, reference = lgb_train)
lgbm_params = {"objective": "multiclass",
               "num_class": 4}
model = lgb.train(lgbm_params, lgb_train, valid_sets=lgb_eval)
del X_train,Y_train, market_train_df, news_train_df
lgb.plot_importance(model, figsize = (5,10))
col_id = {i:v for i,v in enumerate(fcol)}
for i in [10,8,9,7,5,6,7,12,3,1,2,0,11]:
    print(col_id[i])
predicted = model.predict(X_test,num_iteration=model.best_iteration)
y_pred_max = np.argmax(predicted, axis=1)
from sklearn.metrics import accuracy_score
accuracy_score(Y_test,y_pred_max)
Y_test[:5]
predicted[:5]
pd.Series(np.argmax(predicted,axis = 1)).value_counts()
days = env.get_prediction_days()
market_obs_df, news_obs_df, predictions_template_df = next(days)
market_df = data_prep(market_obs_df, news_obs_df)
market_df = market_df[market_df.assetCode.isin(predictions_template_df.assetCode)]
X = market_df[fcol].fillna(0).values
X[X > 1000000000] = 1000000000
X[X < -1000000000] = -1000000000
X = sc.transform(X)
y_pred_prob = model.predict(X, num_iteration=model.best_iteration)
y_pred_category = np.argmax(y_pred_prob, axis = 1)
confidence = (y_pred_category - ((4 - 1) / 2)) / ((4-1)/2)  #4: number of category
preds = pd.DataFrame({"assetCode":market_obs_df["assetCode"],"confidence":confidence})
predictions_template_df = predictions_template_df.merge(preds,how="left")\
    .drop("confidenceValue",axis=1).fillna(0)\
    .rename(columns={"confidence":"confidenceValue"})
predictions_template_df.plot(kind="hist")
env.predict(predictions_template_df)
%%time
for market_obs_df, news_obs_df, predictions_template_df in days:
    market_df = data_prep(market_obs_df, news_obs_df)
    market_df = market_df[market_df.assetCode.isin(predictions_template_df.assetCode)]
    X = market_df[fcol].fillna(0).values
    X[X > 1000000000] = 1000000000
    X[X < -1000000000] = -1000000000
    X = sc.transform(X)
    y_pred_prob = model.predict(X, num_iteration=model.best_iteration)
    y_pred_category = np.argmax(y_pred_prob, axis = 1)
    confidence = (y_pred_category - ((4 - 1) / 2)) / ((4-1)/2)  #4: number of category
    preds = pd.DataFrame({"assetCode":market_obs_df["assetCode"],"confidence":confidence})
    predictions_template_df = predictions_template_df.merge(preds,how="left")\
        .drop("confidenceValue",axis=1).fillna(0)\
        .rename(columns={"confidence":"confidenceValue"})
    env.predict(predictions_template_df)

env.write_submission_file()
import os
print([filename for filename in os.listdir(".") if ".csv" in filename])
