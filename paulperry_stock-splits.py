import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df['date'] = pd.to_datetime(market_train_df.time.dt.date)
apple = market_train_df[market_train_df['assetCode'] == 'AAPL.O']
ax = apple.plot(kind='line',x='date',y='open',figsize=(12,6))
appleNews = news_train_df[news_train_df['assetName'] == 'Apple Inc']
list(appleNews[(appleNews['headline'].str.contains('stock split')) & (appleNews['relevance'] >= 0.6)].head()['headline'])
apple[(apple['time'] > '2014-06-01') & (apple['time'] < '2014-06-16')][['time','close']]
apple['adjOpen'] = np.where(apple['time'] < '2014-06-09',apple['open']/7.0,apple['open'])
apple['MA10'] = apple['adjOpen'].rolling(window=10).mean()
apple['MA50'] = apple['adjOpen'].rolling(window=50).mean()
apple['MA200'] = apple['adjOpen'].rolling(window=200).mean()
ax = apple.plot(kind='line',x='date',y=['adjOpen','MA10','MA50','MA200'], figsize=(16,6))
split_news = news_train_df[(news_train_df['headline'].str.contains('stock split'))  & \
                           (news_train_df['headline'].str.contains('-for-')) & \
                           (news_train_df['relevance'] >= 0.6)]
split_news.shape
split_news['date'] = pd.to_datetime(split_news.time.dt.date)
split_news[['date','headline']].head()
assetCodes = split_news.assetCodes.astype(str).apply(eval).apply(list)
# eliminate irrelevant exchanges
exchanges = {'N', 'O', 'A'} # , 'OQ', 'P', 'OB'} # take only the top exchanges
assetCodes = assetCodes.apply(lambda assetCode: [c for c in assetCode if '.' in c and c.split('.')[1] in exchanges])
assetCodes.apply(len).value_counts()
# just take the first one
assetCodes = assetCodes.apply(lambda x: x[0])
assetCodes.head()
split_news.assetCodes = assetCodes
# remove duplicate news announcements 
split_news.drop_duplicates(subset=['date','assetCodes'], inplace=True)
split_news.shape
split_news[['date','assetCodes','headline']].head(10)
split_news[split_news.assetCodes=='AAPL.O'][['headline']].iloc[0][0]
# make sure the word2number package is installed
# !pip install word2number
from word2number import w2n
split_from = split_news.headline.apply(lambda x: x.split('-for-')[0].rsplit(' ')[-1])
split_from = split_from.apply(w2n.word_to_num)
split_from.value_counts()
split_to = split_news.headline.apply(lambda x: x.split('-for-')[1].rsplit(' ')[0])
split_to = split_to.apply(w2n.word_to_num)
split_to.value_counts()
split_news['split_from'] = split_from
split_news['split_to'] = split_to
split_news[['date','assetCodes','split_from','split_to']].head(10)
# market_train_df[market_train_df.assetCode == 'ZOLL.O']
market_splits = list(set(split_news.assetCodes.unique()).intersection(set(market_train_df.assetCode.unique())))
split_news = split_news[split_news.assetCodes.isin(market_splits)]
split_news.shape
%%time
# determine if the stock split occurs in the market_train date range
for idx, r in split_news.iterrows():
    if market_train_df[(market_train_df.assetCode == r.assetCodes) & (market_train_df.date == r.date)].shape[0] > 0:
        split_news.loc[idx,'traded'] = True
    else:
        split_news.loc[idx,'traded'] = False
split_news[split_news.traded == True].shape
split_news[['time','assetCodes','traded']].head()
market_split_news = split_news[split_news.traded == True]
threshold = 0.2 # + or - window around price differential on the open of the day of the split
for idx, r in market_split_news.iterrows():
    s = market_train_df[(market_train_df.assetCode == r.assetCodes) & 
                    (market_train_df.date > r.date) &
                    (market_train_df.date < r.date+pd.DateOffset(months=3))]
    s = s.set_index('date')
    found_it = False
    prevday = s.index[0]
    for day, r2 in s.iloc[1:].iterrows():  # minus a day
        prevopen = s.iloc[s.index.get_loc(day)-1].open

        if abs(round(r2.returnsOpenPrevRaw1,2) / round(((r2.open-prevopen)/prevopen),2)) < threshold :
            found_it = True
            break
        prevday = day
    if found_it:
        market_split_news.loc[idx,'split_day'] = prevday
    else:
        market_split_news.loc[idx,'split_day'] = pd.NaT

print('found the date to ',market_split_news.split_day.notnull().sum(),' of ',market_split_news.shape[0],' splits.')
for idx, r in market_split_news.iterrows():
    print(r.headline, flush=True)
    s = market_train_df[(market_train_df.assetCode == r.assetCodes) & 
                    (market_train_df.date > r.date-pd.DateOffset(months=1)) &
                    (market_train_df.date < r.date+pd.DateOffset(months=3))]
    s = s.set_index('date')
    s.close.plot(figsize=(12,3))
    plt.title(r.assetCodes+' '+str(r.split_from)+'-for-'+str(r.split_to))
    plt.axvline(r.date,c='g')
    if r.split_day is not pd.NaT:
        plt.axvline(r.split_day,c='r')
    plt.show()
