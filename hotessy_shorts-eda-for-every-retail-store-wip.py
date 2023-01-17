import numpy as np

import pandas as pd 

from pathlib import Path

import os

import warnings

from tqdm.auto import tqdm



warnings.filterwarnings("ignore")

tqdm.pandas()

pd.set_option('display.max_columns', 1000)

pd.set_option('max_rows', 500)

np.random.seed(47)
base_path = Path('/kaggle/input/competitive-data-science-predict-future-sales/')
for a in base_path.iterdir():

    if a.match("*.csv"):

        globals()[a.stem] = pd.read_csv(a)

        print(a.stem)
import chart_studio.plotly as py

import plotly.tools as tls

from plotly.subplots import make_subplots

import plotly.graph_objs as go

import plotly

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

import cufflinks as cf

import plotly.figure_factory as ff

cf.set_config_file(offline=True)
help(sales_train.iplot)
# items['item_category_id'] = items['item_category_id'].astype('str')
items['item_category_id'].iplot(kind='hist')
sales_train['date'] = pd.to_datetime(sales_train['date'], dayfirst=True) 
sales_train_sales = sales_train.query("item_cnt_day > 0")

sales_train_returns = sales_train.query("item_cnt_day < 0")

sales_train['net_sales'] = sales_train['item_price']*sales_train['item_cnt_day']
sales_train.head()
sales_train_sales.shape, sales_train_returns.shape
# Find `sale` corresponding to the `return`



def find_sale(row):

   pass



sales_train_returns.apply(func=find_sale, axis=1)
sales_train.query("shop_id == 25 and item_id == 2552")
sales_train_returns
df = sales_train.groupby('date', as_index=False).agg({'item_price': np.sum, 'shop_id': [pd.Series.nunique], 'item_id': [pd.Series.nunique]})

df.iplot(x='date', subplots=True, shape=(3,1), shared_xaxes=True)

# df.iplot(x='date', y=('item_price', 'sum'), title="('item_price', 'sum')")

# df.iplot(x='date', y=('shop_id', 'nunique'), title="('shop_id', 'nunique')")

# df.iplot(x='date', y=('item_id', 'nunique'), title="('item_id', 'nunique')")
df = sales_train.groupby(['date', 'shop_id'], as_index=False).agg({'item_price': np.sum}).sort_values(['date', 'item_price'])

df['shop_id'] = df['shop_id'].astype(str)

df.iplot(x='date', y='item_price', categories='shop_id', mode='lines')
TOP_K = 10

top_performers = sales_train.groupby(['shop_id'], as_index=False).agg({'item_price': np.sum}).sort_values('item_price').head(TOP_K)['shop_id']

df = sales_train.loc[sales_train['shop_id'].isin(top_performers)].groupby(['date', 'shop_id'], as_index=False).agg({'item_price': np.sum})

assert df['shop_id'].nunique() == TOP_K

df['shop_id'] = df['shop_id'].astype(str)

df.iplot(x='date', y='item_price', categories='shop_id', mode='lines')
df = sales_train.groupby(['shop_id'], as_index=False).agg({'item_price': np.sum})

prcs = [50, 80, 90]

percentiles = np.percentile(df['item_price'], prcs)

# annotations = {percentiles[i]: f"{prc}th Percentile" for i, prc in enumerate(prcs)}

df['shop_id'] = df['shop_id'].astype(str)

# df['item_price'].iplot(kind='hist', bins=100, histnorm='percent', barmode='stack', annotations=annotations)

fig = df['item_price'].figure(kind='hist', bins=100, histnorm='percent', barmode='stack', vline=1e7)
fig['layout']
cf.help('distplot')
from sklearn.preprocessing import MinMaxScaler



scaler = MinMaxScaler()

scaler.fit(df['item_price'].to_numpy().reshape(-1, 1))

data = scaler.transform(df['item_price'].to_numpy().reshape(-1, 1))



ff.create_distplot([data.squeeze(-1).tolist()], ['item_price'], bin_size=0.05)
# ??cf.Figure
# df[['item_price']].iplot(kind='distplot', bins=0.20)

fig = pd.DataFrame(data, columns=['item_price']).figure(kind='distplot', bin_size =0.05, show_curve=True, show_hist=False, show_rug=True)

# cf.Figure(fig)
prcs = [50, 80, 90]

percentiles = np.percentile(data, prcs, interpolation='nearest')

annotations = [{'x': percentiles[i], 'y': fig['data'][0]['y'][(fig['data'][1]['x'] == percentiles[i]).argmax()], 'text': f"{prc}th Percentile"} for i, prc in enumerate(prcs)]

# fig['layout'].update({'annotations': annotations})





vline = go.layout.Shape(type="line", x0=percentiles[0], y0=0, x1=percentiles[0], y1=fig['data'][0]['y'].max(), name='Median')

# fig['layout'].update({'shapes': [vline]})

fig.add_shape(vline)



fig.show()
# fig.update_layout(

#     showlegend=False,

#     annotations=[

#         go.layout.Annotation(

#             x=percentiles[1],

#             y=2,

#             text='bla bla',

#             showarrow=True,

#             arrowhead=5,

#         )

#     ]

# )
df = sales_train.groupby(['shop_id'], as_index=False).agg({'item_price': np.sum})

df = df['item_price'].sort_values(ascending=False).cumsum().reset_index(drop=True)



prc_contri =  df.map(lambda x: x/ df.iloc[-1])

store_index = (prc_contri > 0.80).argmax()



# a1 = {'x': 0.1, 'y': 0.8, 'text': "Pareto Point", 'showarrow': True, 'arrowhead':5, 'xref':"paper", 'yref':"paper", 'textangle':-45, 'ay':100}

a1 = {'x': store_index, 'y': df.iloc[store_index], 

      'text': f"Top {store_index + 1} ({round(((store_index + 1)/df.shape[0])*100, 2)}%) stores contribute to {round(prc_contri[store_index]*100, 2)}% of Total Sales", 

      'showarrow': True, 'arrowhead':5, 'textangle': 0, 'ay':50 }



df.iplot(kind='line', annotations=[a1], xaxis_type='category', xrange=[-1, 61])


# fig = df.iplot(kind='line', asFigure=True)



# fig.update_layout(

#     showlegend=False,

#     annotations=[

#         go.layout.Annotation(

#             x=(df == percentiles[1]).argmax(),

#             y=percentiles[1],

#             text=f"Top {(df == percentiles[1]).argmax()} stores contribute to 80% of Total Sales",

#             showarrow=True,

#             arrowhead=5,

#         )

#     ]

# )
TOP_K = 10

BOTTOM_K = 10



df = sales_train.pivot_table(columns='shop_id', index='date', values='item_price', margins=True, margins_name='Total')

df = df.drop('Total', axis=1)

df.columns = df.columns.astype(str)

top_k_index = df.loc['Total'].sort_values(ascending=False).head(TOP_K).index

bottom_k_index = df.loc['Total'].sort_values(ascending=True).head(BOTTOM_K).index



TOTAL_STORES = df.shape[1]

df.filter(items=top_k_index, axis=1).iplot(kind='box', xaxis_type='category', boxpoints='outliers')

# df.filter(items=bottom_k_index, axis=1).iplot(kind='box')
t_df = df.filter(items=top_k_index, axis=1)

t_df.columns = [str(i + 1) + '. ' + col for i, col in enumerate(t_df.columns)]

t_df.iplot(kind='box', title=f"TOP_K = {TOP_K}", boxpoints='all', xaxis_type='category')
help(cf.subplots)
t_df = df.filter(items=bottom_k_index, axis=1)

t_df.columns = [str(TOTAL_STORES - i) + '. ' + col for i, col in enumerate(t_df.columns)]



fig1 = t_df.figure(kind='box', title=f"BOTTOM_K = {BOTTOM_K}", boxpoints='suspectedoutliers')

fig2 = t_df.figure(kind='box', title=f"BOTTOM_K = {BOTTOM_K}", boxpoints='outliers')

cf.Figure(cf.subplots(figures=[fig1, fig2], shape=(2, 1))).iplot()
shop_item_pivot = sales_train.pivot_table(index='item_id', columns='shop_id', values='item_price', aggfunc=np.sum)
item_rank = shop_item_pivot.rank(na_option='bottom', method='max', ascending=False, axis=0)
shop_rank = shop_item_pivot.rank(na_option='bottom', method='max', ascending=False, axis=1)
item_rank
shop_rank.corrwith(other=item_rank, method='spearman')
shop_item_pivot.corr(method='spearman').iplot(kind='heatmap')