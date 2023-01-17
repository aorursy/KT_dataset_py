# This is my commonly-used standard data science libraries
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# Basic package
import re
import itertools
import datetime
from dateutil import parser
import os
import operator

# DS basic
from scipy import stats
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O


# Viz
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
pd.set_option('max_columns', 50)
#pd.options.display.max_columns = None
plt.style.use('bmh')
color_pal = plt.rcParams['axes.prop_cycle'].by_key()['color']
color_cycle = itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
past_sales = sale_df.set_index('id')[d_cols].T \
    .merge(cal_df.set_index('d')['date'], left_index=True, right_index=True) \
    .set_index('date')
past_sales.head()
## Distribution of sales values
pd.Series((stv[d_cols].values != 0).argmax(axis=1)).hist(figsize=(25, 5), bins=100)



## Sales by items
# Import widgets
from ipywidgets import widgets, interactive, interact
# import ipywidgets as widgets
from IPython.display import display


days = range(1, 1913 + 1)
d_cols = [f'd_{i}' for i in days]

ids = np.random.choice(stv['id'].unique().tolist(), 1000)

series_ids = widgets.Dropdown(
    options=ids,
    value=ids[0],
    description='series_ids:'
)

def plot_data(series_ids):
    df = stv.loc[stv['id'] == series_ids][d_cols]
    df = pd.Series(df.values.flatten())

    df.plot(figsize=(20, 10), lw=2, marker='*')
    df.rolling(7).mean().plot(figsize=(20, 10), lw=2, marker='o', color='orange')
    plt.axhline(df.mean(), lw=3, color='red')
    plt.grid()
    
w = interactive(
    plot_data,
    series_ids=series_ids
)
display(w)


## Total Sales by Categories
# easily using 'stv' but no much info related to real dates. So use 'past_sales'
# stv.groupby('cat_id').sum().T.plot()

for i in stv['cat_id'].unique():
    stv.groupby('cat_id').sum()
    items_col = [c for c in past_sales.columns if i in c]
    past_sales[items_col] \
        .sum(axis=1) \
        .plot(figsize=(15, 5),
              alpha=0.8,
              title='Total Sales by Item Type')
plt.legend(stv['cat_id'].unique())
plt.show()


## Sales broken down by time variables
# - Now that we have our example item lets see how it sells by:
#     - Day of the week
#     - Month
#     - Year

# Merge calendar on our items' data
example = stv.loc[stv['id'] == 'FOODS_3_090_CA_3_validation'][d_cols].T
example = example.rename(columns={8412:'FOODS_3_090_CA_3'}) # Name it correctly
example = example.reset_index().rename(columns={'index': 'd'}) # make the index "d"
example = example.merge(cal, how='left', validate='1:1')
# Select more top selling examples
example2 = stv.loc[stv['id'] == 'HOBBIES_1_234_CA_3_validation'][d_cols].T
example2 = example2.rename(columns={6324:'HOBBIES_1_234_CA_3'}) # Name it correctly
example2 = example2.reset_index().rename(columns={'index': 'd'}) # make the index "d"
example2 = example2.merge(cal, how='left', validate='1:1')

example3 = stv.loc[stv['id'] == 'HOUSEHOLD_1_118_CA_3_validation'][d_cols].T
example3 = example3.rename(columns={6776:'HOUSEHOLD_1_118_CA_3'}) # Name it correctly
example3 = example3.reset_index().rename(columns={'index': 'd'}) # make the index "d"
example3 = example3.merge(cal, how='left', validate='1:1')


examples = ['FOODS_3_090_CA_3','HOBBIES_1_234_CA_3','HOUSEHOLD_1_118_CA_3']
example_df = [example, example2, example3]
for i in [0, 1, 2]:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 3))
    example_df[i].groupby('wday').mean()[examples[i]] \
        .plot(kind='line',
              title='average sale: day of week',
              lw=5,
              color=color_pal[0],
              ax=ax1)
    example_df[i].groupby('month').mean()[examples[i]] \
        .plot(kind='line',
              title='average sale: month',
              lw=5,
              color=color_pal[4],

              ax=ax2)
    example_df[i].groupby('year').mean()[examples[i]] \
        .plot(kind='line',
              lw=5,
              title='average sale: year',
              color=color_pal[2],

              ax=ax3)
    fig.suptitle(f'Trends for item: {examples[i]}',
                 size=20,
                 y=1.1)
    plt.tight_layout()
    plt.show()
    
    

## Sales Price
fig, ax = plt.subplots(figsize=(15, 5))
stores = []
for store, d in sellp.query('item_id == "FOODS_3_090"').groupby('store_id'):
    d.plot(x='wm_yr_wk',
          y='sell_price',
          style='.',
          color=next(color_cycle),
          figsize=(15, 5),
          title='FOODS_3_090 sale price over time',
         ax=ax,
          legend=store)
    stores.append(store)
    plt.legend()
plt.legend(stores)
plt.show()



## Distribution of price

sellp['Category'] = sellp['item_id'].str.split('_', expand=True)[0]
fig, axs = plt.subplots(1, 3, figsize=(15, 4))
i = 0
for cat, d in sellp.groupby('Category'):
    ax = d['sell_price'].apply(np.log1p) \
        .plot(kind='hist',
                         bins=20,
                         title=f'Distribution of {cat} prices',
                         ax=axs[i],
                                         color=next(color_cycle))
    ax.set_xlabel('Log(price)')
    i += 1
plt.tight_layout()





## TODO
# - Simple prediction based on historical average sale by day of week
# - Facebook prophet model
# - lgbm/xgb model based on day features
thirty_day_avg_map = stv.set_index('id')[d_cols[-30:]].mean(axis=1).to_dict()
fcols = [f for f in ss.columns if 'F' in f]
for f in fcols:
    ss[f] = ss['id'].map(thirty_day_avg_map).fillna(0)
    
ss.to_csv('submission.csv', index=False)