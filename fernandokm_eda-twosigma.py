import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from kaggle.competitions import twosigmanews

%matplotlib inline
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()
(market_train_df, news_train_df) = env.get_training_data()
market_train_df.shape
market_train_df.head()
market_missing = pd.DataFrame(market_train_df.isna().sum(), columns=['NA (absolute)'])
market_missing['NA (relative)'] = market_missing['NA (absolute)'] / market_train_df.shape[0]
market_missing
codes = market_train_df['assetCode'].str.split('.', expand=True)
codes[0].value_counts().plot(title='stock (value_counts)')
plt.xlabel('Stock')
plt.ylabel('Count')
plt.show()
codes[1].value_counts().plot(title='exchanges (value_counts)')
plt.xlabel('Exchange')
plt.ylabel('Count')
plt.show()
print("Unique stocks:", len(codes[0].unique()))
print("Unique exchanges:", len(codes[1].unique()))
assets = market_train_df[['assetCode', 'assetName']].copy()
assets['stock'] = assets['assetCode'].apply(lambda code: code[:code.index('.')]) # drop everything after the '.' in assetCode
assets.drop('assetCode', axis=1, inplace=True)
assets.drop(assets[assets['assetName'] == 'Unknown'].index, inplace=True) # drop assets with unknown name
asset_name_counts = len(assets['assetName'].unique())
assets_multiple_stocks = 0
by_name = assets.groupby('assetName', axis=0)
for name in assets['assetName'].unique():
    codes = by_name.get_group(name)['stock'].unique()
    if codes.shape[0] > 1:
        assets_multiple_stocks += 1
print('Total asset name count:', asset_name_counts)
print('Asset names associated with more than one stock:', assets_multiple_stocks)
stock_counts = len(assets['stock'].unique())
stocks_multiple_asset_names = 0
by_stock = assets.groupby('stock', axis=0)
for stock in assets['stock'].unique():
    names = by_stock.get_group(stock)['assetName'].unique()
    if names.shape[0] > 1:
        stocks_multiple_asset_names += 1
print('Total stock count:', stock_counts)
print('Stocks associated with more than one asset name:', stocks_multiple_asset_names)
assets['assetName'].value_counts().plot(title='assetName (value_counts)')
plt.xlabel('assetName')
plt.ylabel('Count')
plt.show()
print("Unique names:", len(assets['assetName'].unique()))
market_train_df['universe'].value_counts()
universe_per_time = pd.Series(sum(market_train_df['universe'][group])/group.shape[0] for group in market_train_df.groupby('time')['universe'].groups.values())
universe_per_time.describe()
market_train_df.plot(x='time', y='volume')
market_train_df.groupby('time')[['open', 'close']].mean().plot(title='Average open and close prices per day', figsize=(20, 10))
market_train_df['change'] = market_train_df['close'] - market_train_df['open']
market_train_df['relative_change'] = market_train_df['change'] / market_train_df['open']
print('Statistics for the relative change in stock price per day')
market_train_df.groupby('time')['relative_change'].mean().describe()

market_train_df.groupby('time')['change'].mean().plot(title='Average change in price per day', figsize=(20, 10))
plt.show()
market_train_df.groupby('time')['relative_change'].mean().plot(title='Average relative change in price per day', figsize=(20, 10))
plt.show()
market_train_df.groupby('time')[['returnsClosePrevRaw1', 'returnsClosePrevMktres1']].mean().plot(figsize=(15,7))
plt.show()
market_train_df.groupby('time')[['returnsOpenPrevRaw1', 'returnsOpenPrevMktres1']].mean().plot(figsize=(15,7))
plt.show()
market_train_df.groupby('time')[['returnsClosePrevRaw10', 'returnsClosePrevMktres10']].mean().plot(figsize=(15,7))
plt.show()
market_train_df.groupby('time')[['returnsOpenPrevRaw10', 'returnsOpenPrevMktres10']].mean().plot(figsize=(15,7))
plt.show()
sns.pairplot(data=market_train_df.groupby('time').mean()[['volume', 'open', 'close', 'returnsClosePrevRaw10', 'returnsOpenNextMktres10']])
news_train_df.shape
news_train_df.head()
