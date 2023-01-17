import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import datetime
from sklearn.metrics import accuracy_score
from kaggle.competitions import twosigmanews
import gc
env = twosigmanews.make_env()
(market_train, news_train) = env.get_training_data()
gc.enable()
import warnings
import datetime
warnings.filterwarnings(action ='ignore',category = DeprecationWarning)

news_train['time'] = news_train['time'].dt.floor('d')

cols = ['sentimentNegative','sentimentNeutral','sentimentPositive','relevance','companyCount','bodySize','sentenceCount','wordCount','firstMentionSentence']
def get_news_train(raw_data,days = 2):
    news_last = pd.DataFrame()
    for i in range(days):
        cur_train = raw_data[cols]
        cur_train['time'] = raw_data['time'] + datetime.timedelta(days = i,hours=22)
        cur_train['key'] = cur_train['time'].astype(str)+ raw_data['assetName'].astype(str)
        news_last = pd.concat([news_last, cur_train[['key'] + cols]])
        print("after concat the shape is:",news_last.shape)
        news_last = news_last.groupby('key').sum()
        news_last['key'] = news_last.index.values
        print("the result shape is:",news_last.shape)
        del cur_train
        gc.collect()
    del news_last['key']
    return news_last

news_last = get_news_train(news_train)
print(news_last.shape)
print(news_last.head())
print(news_last.dtypes)

market_train['key'] = market_train['time'].astype(str) + market_train['assetName'].astype(str)
market_train = market_train.join(news_last,on = 'key',how='left')
print(market_train['sentimentNeutral'].isnull().value_counts())
market_train.head()