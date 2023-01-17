import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
from itertools import chain
from kaggle.competitions import twosigmanews

env = twosigmanews.make_env()
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
print(f'market train df shape: {market_train_df.shape}')
print(f'news train df shape: {news_train_df.shape}')
market_train_df.head()
market_train_df.info()
missing_count = market_train_df.isna().sum()
missing_count
plt.figure(figsize=(12,8))
plt.bar(missing_count.index, missing_count.values)
plt.xticks(rotation=45)
plt.show()
# show number of missing values over time

missing_col = ['returnsClosePrevMktres1', 'returnsOpenPrevMktres1', 
               'returnsClosePrevMktres10', 'returnsOpenPrevMktres10']
df_na = market_train_df[market_train_df.isnull().any(axis=1)]
missing_day = df_na.loc[:, missing_col].isnull().groupby(df_na.time).sum()

f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, figsize=(12,8))
ax1.plot(missing_day.index, missing_day.returnsOpenPrevMktres1)
ax1.set_ylim(0,100)
ax1.set_title('returnsClosePrevMktres1')
ax2.plot(missing_day.index, missing_day.returnsOpenPrevMktres1)
ax2.set_ylim(0,100)
ax2.set_title('returnsOpenPrevMktres1')
ax3.plot(missing_day.index, missing_day.returnsClosePrevMktres10)
ax3.set_ylim(0,200)
ax3.set_title('returnsClosePrevMktres10')
ax4.plot(missing_day.index, missing_day.returnsOpenPrevMktres10)
ax4.set_ylim(0,200)
ax4.set_title('returnsOpenPrevMktres10')
plt.show()
print(f'number of unique asset Codes: {market_train_df.assetCode.unique().shape[0]}')
print(f'number of unique asset Names: {market_train_df.assetName.unique().shape[0]}')
market_train_df.describe()
# histograme for log10(volume)

plt.hist(market_train_df.volume.apply(lambda x: np.log10(x) if x!=0 else 0), bins=50)
plt.title('volume')
plt.show()
# there are some very high open(10k) and close value(1.5k) 
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(10,4))
ax1.hist(market_train_df.open, bins=50, range=(0,500))
ax1.set_title('open')
ax2.hist(market_train_df.close, bins=50, range=(0,500))
ax2.set_title('close')
plt.show()
f, axes = plt.subplots(3, 3, sharey=True, figsize=(15,15))
for i in range(3):
    for j in range(3):
        axes[i,j].hist(market_train_df.iloc[:, 6+i*3+j], range=(-0.5,0.5), bins=50)
        axes[i,j].set_title(market_train_df.columns[6+i*3+j])
plt.show()
news_train_df.head()
news_train_df.info()
# no missing value here
news_train_df.isna().sum().sum()
news_train_df.describe()
# plot histogram of all the numeric columns
news_train_df.select_dtypes(include=[np.number]).hist(figsize=(15,15))
plt.show()
n_codes = len(set(chain(*news_train_df['assetCodes'].str.findall(f"'([\w\./]+)'"))))
print(f'number of unique asset Codes in news set: {n_codes}')
print(f'number of unique asset Names in news set: {news_train_df.assetName.unique().shape[0]}')
print('*'*50)
print(f'number of unique asset Codes in market set: {market_train_df.assetCode.unique().shape[0]}')
print(f'number of unique asset Names in market set: {market_train_df.assetName.unique().shape[0]}')
days = env.get_prediction_days()
(market_obs_df, news_obs_df, predictions_template_df) = next(days)
print(f'market_obs_df shape: {market_obs_df.shape}')
print(f'news_obs_df shape: {news_obs_df.shape}')
print(f'predictions_template_df shape: {predictions_template_df.shape}')
market_obs_df.head()
news_obs_df.head()
print(f'date in market set: {market_obs_df.time.dt.date.unique().tolist()}')
print(f'date in news set: {news_obs_df.time.dt.date.unique().tolist()}')