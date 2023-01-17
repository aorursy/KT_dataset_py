# Import libraries
from kaggle.competitions import twosigmanews # official library for comp. 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # fancy plotting 
from pylab import subplot #
import matplotlib.pyplot as plt #

import plotly.offline as py # offline = free plotly (haven't won the comp, yet...)
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
# Create virtual enviornment specific for competition
env = twosigmanews.make_env()
# Extract composed training data 
(market_train_df, news_train_df) = env.get_training_data()
print(f'{market_train_df.shape[0]} samples and {market_train_df.shape[1]} features in the training market dataset.')
# Explore structure of training data
market_train_df.head()
# Explore structure of training data
market_train_df.describe()
#market_train_df.std()
data = []
for asset in np.random.choice(market_train_df['assetName'].unique(), 10):
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]

    data.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Closing prices of 10 random assets",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data=data, layout=layout), filename='basic-line')
sns.set(style="white")
corr = market_train_df.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.show()
# Create new feature "return"
market_train_df.insert(loc=6, column='return', value=market_train_df['close'] / market_train_df['open'])
# Find "suspicious" return 
market_train_df.sort_values('return')[:20]
# Opens ending in XXX.99 seems likely to be false
market_train_df[market_train_df['return'] > 20]