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
print('Done!')
(market_train_df, news_train_df) = env.get_training_data()
import matplotlib.pyplot as plt
import datetime
import plotly.graph_objs as go
import plotly.offline as py
from py import *
py.offline.init_notebook_mode(connected=True)
price = []
target_assetName=["3M Co","American Express Co","Apple Inc","Boeing Co","Caterpillar Inc","Chevron Corp","Cisco Systems Inc","Coca-Cola Co","Walt Disney Co","Exxon Mobil Corp","Goldman Sachs Group Inc","Home Depot Inc","International Business Machines Corp","Intel Corp","Johnson & Johnson","JPMorgan Chase & Co","McDonald's Corp","Microsoft Corp","Nike Inc","Pfizer Inc","Procter & Gamble Co","Travelers Companies Inc","United Technologies Corp","UnitedHealth Group Inc","Verizon Communications Inc","Visa Inc","Walmart Inc","Walgreen Co"]
target_assetCode=["MMM.N","AXP.N","AAPL.O","BA.N","CAT.N","CVX.N","CSCO.O","KO.N","DIS.N","XOM.N","GS.N","HD.N","IBM.N","INTC.O","JNJ.N","JPM.N","MCD.N","MSFT.O","NKE.N","PFE.N","PG.N","TRV.N","UTX.N","UNH.N","VZ.N","V.N","WMT.N","WAG.N"]
asset_picked = []
for asset in np.random.choice(target_assetName,10,replace = False):
    asset_picked.append(asset)
    asset_df = market_train_df[(market_train_df['assetName'] == asset)]
    price.append(go.Scatter(
        x = asset_df['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df['close'].values,
        name = asset
    ))
layout = go.Layout(dict(title = "Dow Jones Components",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Price (USD)'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data = price, layout = layout), filename='Price')
excess_return = []
for asset in asset_picked:
    asset_df2 = market_train_df[(market_train_df['assetName'] == asset)]
    excess_return.append(go.Scatter(
        x = asset_df2['time'].dt.strftime(date_format='%Y-%m-%d').values,
        y = asset_df2['returnsOpenNextMktres10'].values,
        name = asset))
layout = go.Layout(dict(title = "Dow Jones Components",
                  xaxis = dict(title = 'Month'),
                  yaxis = dict(title = 'Excess Return'),
                  ),legend=dict(
                orientation="h"))
py.iplot(dict(data = excess_return, layout = layout), filename='return')
asset_df3 = pd.DataFrame()
for asset in asset_picked:
    asset_df3=asset_df3.append(market_train_df[(market_train_df['assetName'] == asset)],ignore_index=True)
asset_df3['price_diff'] = asset_df3['close'] - asset_df3['open']
grouped = asset_df3.groupby('time').agg({'price_diff': ['std', 'min']}).reset_index()
print(grouped)
print(f"Average standard deviation of price change within a day in {grouped['price_diff']['std'].mean():.4f}.")
g = grouped.sort_values(('price_diff', 'std'), ascending=False)[:10]
g['min_text'] = 'Maximum price drop: ' + (-1 * g['price_diff']['min']).astype(str)
trace = go.Scatter(
    x = g['time'].dt.strftime(date_format='%Y-%m-%d').values,
    y = g['price_diff']['std'].values,
    mode='markers',
    marker=dict(
        size = g['price_diff']['std'].values,
        color = g['price_diff']['std'].values,
        colorscale='Portland',
        showscale=True
    ),
    text = g['min_text'].values
    #text = f"Maximum price drop: {g['price_diff']['min'].values}"
    #g['time'].dt.strftime(date_format='%Y-%m-%d').values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'Top 10 months by standard deviation of price change within a day',
    hovermode= 'closest',
    yaxis=dict(
        title= 'price_diff',
        ticklen= 5,
        gridwidth= 2,
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='sd')