# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import datetime
import gc
import time
import warnings
from itertools import chain

import lightgbm as lgb
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from matplotlib import pyplot as plt
from sklearn.ensemble import IsolationForest, VotingClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
warnings.filterwarnings("ignore")
import os
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(marketdata, news) = env.get_training_data()
marketdata.head(1)
marketdata["time"] = marketdata["time"].dt.date
marketdata.rename(columns={"time": "date"}, inplace=True)
marketdata.shape
len(marketdata[marketdata.assetCode=='A.N']) #Number of Days
len(news.time) #number of news
marketdata.describe()
marketdata.isna().sum()
marketdata["returnsClosePrevMktres1"].fillna(marketdata["returnsClosePrevRaw1"], inplace=True)
marketdata["returnsOpenPrevMktres1"].fillna(marketdata["returnsOpenPrevRaw1"], inplace=True)
marketdata["returnsClosePrevMktres10"].fillna(marketdata["returnsClosePrevRaw10"], inplace=True)
marketdata["returnsOpenPrevMktres10"].fillna(marketdata["returnsOpenPrevRaw10"], inplace=True)
print(marketdata.isna().sum())
returns = marketdata["close"].values / marketdata["open"].values
outliers = ((returns > 1.5).astype(int) + (returns < 0.5).astype(int)).astype(bool)
marketdata = marketdata.loc[~outliers, :]
marketdata.shape
marketdata.describe()
marketdata.sort_values('returnsOpenPrevRaw1',ascending=False)[:5]
#We can check for the stocks above and the days above
marketdata[(marketdata.assetCode=='EXH.N') & (marketdata.date==pd.to_datetime("2007-8-23").date())]
return_columns = ['returnsClosePrevRaw1', 'returnsOpenPrevRaw1', 'returnsClosePrevMktres1', 'returnsOpenPrevMktres1','returnsClosePrevRaw10', 'returnsOpenPrevRaw10', 'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
for i in return_columns:
    returns = marketdata[i].values
    outliers = ((returns > 1).astype(int) + (returns < -0.7).astype(int)).astype(bool)
    marketdata = marketdata.loc[~outliers, :]
marketdata.shape
del returns
del outliers
marketdata.describe()
marketdata = marketdata.loc[marketdata["assetName"] != "Unknown", :]
plt.figure(figsize=(10,5))
plt.plot(marketdata.groupby('date').returnsOpenPrevRaw10.mean().index,marketdata.groupby('date').returnsOpenPrevRaw10.mean().values,color='green')
plt.title("Mean 10 Day Returns")
plt.show()

plt.figure(figsize=(10,5))
plt.plot(marketdata.groupby('date').returnsOpenPrevRaw1.mean().index,marketdata.groupby('date').returnsOpenPrevRaw1.mean().values,color='brown')
plt.title("Mean 1 Day Returns")
plt.show()
for asset in np.random.choice(marketdata['assetName'].unique(), 5):
    asset_df = marketdata[(marketdata['assetName'] == asset)]
    plt.figure(figsize=(10,5))
    plt.plot(asset_df.date,asset_df.returnsOpenPrevRaw1,color="blue")
    plt.title(asset)
    plt.show()
del asset_df
gc.collect()
news.isna().sum() # no NA
news.sample(2)
news["sourceTimestamp"] = news["sourceTimestamp"].dt.date #Convert time to date
news.rename(columns={"sourceTimestamp": "date"}, inplace=True) #Rename accurately
#Normalize the location of the mentioning in word and sentence counts
news["realfirstMentionPos"] = news["firstMentionSentence"].values / news["sentenceCount"].values
news["realSentimentWordCount"] = news["sentimentWordCount"].values / news["wordCount"].values
#Normalization Continues
news["realSentenceCount"] = news.groupby(["date"])["sentenceCount"].transform(lambda x: (x - x.mean()) / x.std())
news["realWordCount"] = news.groupby(["date"])["wordCount"].transform(lambda x: (x - x.mean()) / x.std())
news["realBodySize"] = news.groupby(["date"])["bodySize"].transform(lambda x: (x - x.mean()) / x.std())
