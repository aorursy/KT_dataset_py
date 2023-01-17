import pandas as pd

import numpy as np

import plotly.offline as plo

import plotly.graph_objs as go

import matplotlib.pyplot as plt
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname,filename))
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv', index_col='date')

items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

items_cat = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv', index_col='shop_id')

sales_train.index = pd.to_datetime(sales_train.index, format="%d.%m.%Y")
sales_train.head()
sales_train.info()
sales_train.isna().sum()
items.info()
items_cat.info()
shops.info()
# sales_train = sales_train.drop(columns={

#     'date_block_num'

# }, axis=1

# )
sales_train.head(3)
sales_train.loc[:,['item_price']].plot(figsize=(20,8))
# temp = sales_train.loc[:,['item_price']]

# data = go.Scatter(

#     x = temp.index,

#     y = temp.values,

#     mode = 'markers+lines',

#     name = 'item_price'

# )

# layout = go.Layout(

#     title = 'Item price graph',

#     template = 'plotly_dark'

# )

# fig = go.Figure(data=data, layout=layout)

# plo.iplot(fig)
sales_train
sales_train.reset_index().groupby(by=['date_block_num','shop_id','item_id']).mean()
aa = sales_train.groupby(by=['date_block_num','shop_id','item_id']).agg(

    {

        "item_price" : ['mean'],

        "item_cnt_day" : ['sum']

    }

)

aa.head()
aa.loc[:,'item_price'].plot(figsize=(20,8))
sales_train.groupby(by=['item_id','shop_id']).mean()
ts = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum()
data = go.Scatter(

    x = ts.index,

    y = ts.values,

    mode = 'markers+lines',

    marker = dict(color='green')

)

layout = go.Layout(

    template='plotly_dark'

)

fig = go.Figure(data=data, layout=layout)

plo.iplot(fig)
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf
fig,ax = plt.subplots(figsize=(20,8))

plot_acf(ts, ax = ax,lags=15);
mm = sales_train.groupby(['date'])['item_cnt_day'].sum()
mm.plot(figsize=(20,8))
m = seasonal_decompose(mm)

fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(30,30))

m.trend.plot(ax=ax1)

m.seasonal.plot(ax=ax2)

m.resid.plot(ax=ax3)