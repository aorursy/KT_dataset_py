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
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

(market_train_df, news_train_df) = env.get_training_data()
market_train_df.dtypes
market_train_df.tail()
market_train_df.isna().sum()
import matplotlib as plt
((market_train_df.isnull().sum()/market_train_df.shape[0])*100).sort_values(ascending=False).plot(kind='bar')
market_train_df.nunique()

print("Min date: ",market_train_df['time'].min())
print("Max date: ",market_train_df['time'].max())
market_train_df['time'].dt.time.describe()
market_train_df[['assetCode','assetName']].head()
print("unique asset name", market_train_df['assetName'].nunique())
print("unique asset code", market_train_df['assetCode'].nunique())
print("Difference", abs(market_train_df['assetName'].nunique()-market_train_df['assetCode'].nunique()))

market_train_df[market_train_df['assetName']=='Unknown'].head()
unknown_assetname_codes=market_train_df[market_train_df['assetName']=='Unknown']['assetCode'].unique()
unknown_assetname_codes
import numpy as np
for code in unknown_assetname_codes:
   print(np.count_nonzero(market_train_df[market_train_df['assetCode']==code]['assetName']!='Unknown'))
market_train_df[market_train_df['assetCode']=='AAPL.O']
market_train_df[market_train_df['assetCode']=='A.N']
def volume_trend(assetCode):
    market_train_df[market_train_df['assetCode']==assetCode].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby(['time','assetCode']).sum().plot(kind='line',figsize=(25,5))

volume_trend('AAPL.O')
top_10_byvolume=market_train_df[(market_train_df['time'].dt.year==2016)&(market_train_df['time'].dt.day==30)&(market_train_df['time'].dt.month==12)].sort_values(by='volume',ascending=False)[['assetCode','volume']].head(10)
import matplotlib.pyplot as plt

market_train_df[market_train_df['assetCode'].isin(list(top_10_byvolume['assetCode']))].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby(['time','assetCode']).sum().unstack().plot(figsize=(25,10))
#plot(x='time', y='volume')
#market_train_df[market_train_df['assetCode']==assetCode].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby(['time','assetCode']).sum().plot(kind='line',figsize=(25,5))
#some of the stocks have data for few years
market_train_df[market_train_df['assetCode']=='AMD.O']['time'].dt.year.unique()
market_train_df[market_train_df['assetCode'].isin(list(top_10_byvolume['assetCode']))].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='line')
market_train_df[market_train_df['assetCode']=='AAPL.O'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='line')

market_train_df[market_train_df['assetCode']=='AAPL.O'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='bar')

market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='line')

market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='bar')

market_train_df[market_train_df['assetCode'].isin(list(top_10_byvolume['assetCode']))].sort_values(by='time',ascending=True)[['time','assetCode','volume']].groupby([market_train_df['time'].dt.year,'assetCode']).mean().unstack().plot(figsize=(25,10),kind='line')
market_train_df['volume'].describe()
market_train_df['close'].describe()
market_train_df[market_train_df['assetCode']=='BAC.N'].head()
market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','close']].groupby('time').median().plot(figsize=(25,10),kind='line')
market_train_df[market_train_df['assetCode']=='BAC.N'].sort_values(by='time',ascending=True)[['time','close']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year]).median().plot(figsize=(25,10),kind='line')
import random
num_to_select = 5                          # set the number to select here.
list_of_random_assets = random.sample(list(set(market_train_df['assetCode'])), num_to_select)
list_of_random_assets 
market_train_df[market_train_df['assetCode'].isin(list(list_of_random_assets))].sort_values(by='time',ascending=True)[['time','assetCode','close']].groupby(['time','assetCode']).median().unstack().plot(figsize=(25,10))

list(set(market_train_df[market_train_df['assetCode']=='SGI.O']['time'].dt.year))
market_train_df[market_train_df['assetCode'].isin(list(list_of_random_assets))].sort_values(by='time',ascending=True)[['time','assetCode','close']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10))

market_train_df[market_train_df['assetCode'].isin(list(list_of_random_assets))].sort_values(by='time',ascending=True)[['time','assetCode','close']].groupby([market_train_df['time'].dt.month,market_train_df['time'].dt.year,'assetCode']).median().unstack().plot(figsize=(25,10),kind='bar')

desc_assets=market_train_df[market_train_df['time'].dt.year==2016].groupby('assetCode').describe()
desc_assets
#list(top_10_byvolume['assetCode'])
desc_assets['close'].transpose()[list(top_10_byvolume['assetCode'])].boxplot(figsize=(25,10))

market_train_df[market_train_df['assetCode']=='BAC.N'][['time','assetCode','volume','close','open']].head()
#market_train_df[market_train_df['assetCode']=='BAC.N'][['time','assetCode','volume','close','open']]