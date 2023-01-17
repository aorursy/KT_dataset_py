import numpy as np
import pandas as pd
import random
from kaggle.competitions import twosigmanews
import matplotlib.pyplot as plt
%matplotlib inline
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.columns
market_train_df['Value'] = market_train_df['close'] - market_train_df['open']
market_train_df[['time', 'assetCode', 'volume', 'close', 'open',
       'returnsClosePrevRaw1', 'returnsOpenPrevRaw1',
       'returnsClosePrevMktres1', 'returnsOpenPrevMktres1',
       'returnsClosePrevRaw10', 'returnsOpenPrevRaw10',
       'returnsClosePrevMktres10', 'returnsOpenPrevMktres10',
       'returnsOpenNextMktres10', 'universe']]
for company in random.choices(market_train_df['assetCode'].unique(),k=3):
    print(market_train_df[market_train_df['assetCode']==company])
print(pd.Series(market_train_df['assetName'].unique()))
print(len(pd.Series(market_train_df['assetName'].unique())))

plt.figure(figsize=(20,10))
plt.plot(market_train_df[market_train_df['assetCode'] == 'A.N']['Value'])
for column in market_train_df.columns:
    print('number of unique values in ',column,':',market_train_df[column].nunique())
market_train_df.shape
news_train_df.head()
news_train_df.nunique()