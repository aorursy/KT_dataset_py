from kaggle.competitions import twosigmanews
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') # nice theme for plotting

env = twosigmanews.make_env() # load env
market_data = env.get_training_data()[0] # load only market data
# save the dates for later purpose
dates = pd.to_datetime(market_data['time'].unique())
import matplotlib.pyplot as plt
market_data.groupby('time').count()['universe'].plot(figsize=(12, 5) ,linewidth=2)
print("There are {} unique investable assets in the whole history.".format(market_data['assetName'].unique().shape[0]))
close_price_df = market_data.pivot_table(index='time', values='close', columns='assetCode')
close_price_df['AAPL.O'].plot(linewidth=2, figsize=(12, 5))
ticker = 'AAPL.O'
close_1d_returns_raw = market_data.pivot_table(index='time', values='returnsClosePrevRaw1', columns='assetCode')
close_1d_returns_adj = market_data.pivot_table(index='time', values='returnsClosePrevMktres1', columns='assetCode')
tmp_r = pd.concat([close_1d_returns_raw[ticker],close_1d_returns_adj[ticker]], 1)
tmp_r.columns = ['1 day close-to-close raw returns'.format(ticker), '1 day close-to-close market residualised {} returns'.format(ticker)]
tmp_r.plot(linewidth=1, alpha=0.7, figsize=(12, 5))
import numpy as np
r = tmp_r.iloc[:, 0]
cum_d = (np.cumprod(1 + r) - 1)
cum_l = r.cumsum()
cum_ret = pd.concat([cum_d, cum_l], 1)
cum_ret.columns = ['pct change {} returns'.format(ticker), 'log {} returns'.format(ticker)]
cum_ret.plot(linewidth=2, alpha=1, figsize=(12, 5))
