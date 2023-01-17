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
import seaborn as sns
%matplotlib inline

import datetime

from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()
print('done!')
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.head()
market_train_df.isnull().sum()
market_train_df['returnsClosePrevMktres1'] = market_train_df['returnsClosePrevMktres1'].fillna(market_train_df['returnsClosePrevMktres1'].mean())
market_train_df['returnsOpenPrevMktres1'] = market_train_df['returnsOpenPrevMktres1'].fillna(market_train_df['returnsOpenPrevMktres1'].mean())
market_train_df['returnsClosePrevMktres10'] = market_train_df['returnsClosePrevMktres10'].fillna(market_train_df['returnsClosePrevMktres10'].mean())
market_train_df['returnsOpenPrevMktres10'] = market_train_df['returnsOpenPrevMktres10'].fillna(market_train_df['returnsOpenPrevMktres10'].mean())
market_train_df.isnull().sum()
print(market_train_df['time'].describe())
print(market_train_df['assetCode'].describe())
print(2498 * 3780)
data = pd.DataFrame()
market_train_df['Fluctuation'] = market_train_df['open'] - market_train_df['close']
for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]
    data = data.append(asset_df)
   
fig, ax = plt.subplots(figsize=(15, 10))
lineplot = sns.lineplot(x="time", y="close", data=data,hue='assetCode',ax=ax)

fig, ax = plt.subplots(figsize=(15, 10))
lineplot = sns.lineplot(x="time", y="Fluctuation", data=data,hue='assetCode',ax=ax)
sns.pairplot(data.loc[:,['Fluctuation','volume','assetCode']],hue='assetCode', height=3,aspect=3)
sns.pairplot(data.loc[:,['Fluctuation',
                         'returnsClosePrevRaw1',
                         'returnsOpenPrevRaw1',
                         'assetCode']],hue='assetCode', height=3,aspect=3)
sns.pairplot(data.loc[:,['Fluctuation',
                         'returnsClosePrevMktres1',
                         'returnsOpenPrevMktres1',
                         'assetCode']],hue='assetCode', height=3,aspect=3)
sns.pairplot(data.loc[:,['Fluctuation',
                         'returnsClosePrevRaw10',
                         'returnsOpenPrevRaw10',
                         'assetCode']],hue='assetCode', height=3,aspect=3)
sns.pairplot(data.loc[:,['Fluctuation',
                         'returnsOpenNextMktres10',
                         'assetCode']],hue='assetCode', height=3,aspect=3)