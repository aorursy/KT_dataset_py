import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

# clustering
from sklearn.cluster import KMeans

# time
from pandas.tseries.holiday import USFederalHolidayCalendar
# from sklearn.preprocessing import LabelEncoder
import datetime

# training
from sklearn.model_selection import train_test_split
# import lightgbm as lgb

# import environment for data
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
# Any results you write to the current directory are saved as output.
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
news_train_df.head()
market_train_df.shape
news_train_df.shape
news_train_df["time"] = pd.to_datetime(news_train_df["time"],infer_datetime_format=True)
print("time")
news_train_df["sourceTimestamp"] = pd.to_datetime(news_train_df["sourceTimestamp"],infer_datetime_format=True)
print("sourceTimestamp")
news_train_df["firstCreated"] = pd.to_datetime(news_train_df["firstCreated"],infer_datetime_format=True)
print("firstCreated")
news_train_df.dtypes
## For now remove universe 0 rows. We could use the mfor context later though
# returnsOpenNextMktres10
print(market_train_df.shape[0])
market_train_df = market_train_df.loc[market_train_df.universe>0]
print(market_train_df.shape[0])
market_assetName = set(market_train_df.assetName)
print(len(market_assetName))
news_train_df.head()
print("orig news shape:",news_train_df.shape[0])
news_train_df = news_train_df.loc[news_train_df.assetName.isin(market_assetName)]
news_train_df.shape[0]
market_train_df.tail()
market_train_df.drop(['universe'],axis=1,inplace=True)
market_train_df.columns
market_train_df.tail()
market_train_df["close_open_diff"] = market_train_df["close"]/market_train_df["open"]
market_train_df["volume_money_mean"] = (market_train_df["close"]*market_train_df["volume"] + market_train_df["open"]*market_train_df["volume"])/2

## Add extra target col - binary
market_train_df["binary_returnsNextMktres10"] = market_train_df["returnsOpenNextMktres10"]>0

## from : https://www.kaggle.com/magichanics/amateur-hour-using-headlines-to-predict-stocks
def clustering(df):

    def cluster_modelling(features):
        df_set = df[features]
        cluster_model = KMeans(n_clusters = 9)
        cluster_model.fit(df_set)
        return cluster_model.predict(df_set)
    
    # get columns:
    vol_cols = [f for f in df.columns if f != 'volume' and 'volume' in f]
    novelty_cols = [f for f in df.columns if 'novelty' in f]
    
#     prev_returns_cols = ["returnsClosePrevMktres1","returnsOpenPrevRaw1","returnsClosePrevMktres10"] # mktRes have NaNs! 
    prev_returns_cols =["returnsOpenPrevRaw1","returnsClosePrevRaw10","returnsOpenPrevRaw10"]
    
    # fill nulls
    cluster_cols = novelty_cols + vol_cols + ['open', 'close']
    df[cluster_cols] = df[cluster_cols].fillna(0)
    
    df['cluster_open_close'] = cluster_modelling(['open', 'close'])
    df['cluster_volume'] = cluster_modelling(vol_cols)
#     df['cluster_novelty'] = cluster_modelling(novelty_cols)
    df['cluster_prev_returns'] = cluster_modelling(prev_returns_cols)
    
    return df
market_train_df = clustering(market_train_df)
market_train_df.tail(3)

market_train_df.to_csv("market_train_uni1_v1.csv.gz",index=False,compression="gzip")
news_train_df.to_csv("news_market_train_uni1_v1.csv.gz",index=False,compression="gzip")