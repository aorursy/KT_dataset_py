import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
print('length of market_train_df: {}'.format(len(market_train_df)))
print('how many assetCode in training set: {}'.format(market_train_df['assetCode'].nunique()))
print('length of news_train_df: {}'.format(len(news_train_df)))
print(market_train_df.columns)
market_train_df.head(2)
print(market_train_df.columns)
market_train_df.head(2)
for i in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    market_train_df.groupby('time')['close'].quantile(i).plot(figsize=(10,6))
market_train_df['close_to_open_abs'] = (market_train_df['close'] / market_train_df['open']).abs()
print('price increased by 100% or more: {}'.format(len(market_train_df[market_train_df['close_to_open_abs'] >= 2])))
print('price decreased by 100% or more: {}'.format(len(market_train_df[market_train_df['close_to_open_abs'] <= 0.5])))
# 얘네를 어떤 값으로 대체를 해줘야 좋을까?
market_train_df[(market_train_df['close_to_open_abs'] >= 2) | (market_train_df['close_to_open_abs'] <= 0.5)]
print(news_train_df.columns)
news_train_df.head(2)