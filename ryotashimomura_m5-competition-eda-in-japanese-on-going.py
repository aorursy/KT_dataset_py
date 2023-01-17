import datetime
import gc
import random
import warnings
from typing import List
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.io as pio; pio.renderers.default='notebook'
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
from IPython.display import display

# from src.notebook_utils import reduce_memory_usage, merge_by_concat, show_basic_info
# from src.df_transformer import ReplaceBeforeFirstSell
# from src.plot_utils import PlotlyFigure
def reduce_memory_usage(
    df: pd.DataFrame,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    受け取ったデータフレームについてそれぞれのカラムのデータ型を調べ、最小メモリ型に変換することで
    データフレームのメモリ使用量を削減する。
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and \
                   c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and \
                        c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and \
                        c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and \
                        c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and \
                   c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and \
                        c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            'Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'
            .format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def merge_by_concat(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    merge_on: List[str],
) -> pd.DataFrame:
    """
    型情報を失わずにデータフレームをマージする
    """
    merged_gf = df1[merge_on]
    merged_gf = merged_gf.merge(df2, on=merge_on, how='left')
    new_columns = [col for col in list(merged_gf) if col not in merge_on]
    df1 = pd.concat([df1, merged_gf[new_columns]], axis=1)
    return df1


def show_basic_info(df: pd.DataFrame):
    """
    データフレームの基本情報を表示する
    """
    df.info()
    display(df.head())
    display(df.tail())
class ReplaceBeforeFirstSell(object):

    def __init__(self):
        pass

    def transform(
        self,
        df: pd.DataFrame,
        value_columns: List[str],
    ) -> pd.DataFrame:
        df_values = df[value_columns].values
        tmp = np.tile(np.arange(1, len(value_columns) + 1),
                      (df_values.shape[0], 1))
        tmp_values = ((df_values > 0) * tmp)
        start_no = np.min(np.where(
            tmp_values == 0, 9999, tmp_values), axis=1) - 1
        flag = np.dot(np.diag(1 / (start_no + 1)), tmp) < 1
        df_values = np.where(flag, np.nan, df_values)
        df[value_columns] = df_values
        return df
class PlotlyFigure(object):
    """
    Provide application methods as adapter class of plotly
    """
    def __init__(self):
        pass

    def add_range_selector(
        self,
        fig: go.Figure,
    ) -> go.Figure:
        fig.update_xaxes(
            rangeselector=dict(
                buttons=list([
                    dict(count=12*7, label="12w",
                         step="day", stepmode="backward"),
                    dict(count=24*7, label="24w",
                         step="day", stepmode="backward"),
                    dict(count=36*7, label="36w",
                         step="day", stepmode="backward"),
                    dict(count=1, label="1y",
                         step="year", stepmode="backward"),
                    dict(count=2, label="2y",
                         step="year", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )
        return fig

    def add_shape_region(
        self,
        fig: go.Figure,
        start_date: str,
        end_date: str,
        color: str = None,
    ) -> go.Figure:
        fig.add_shape(
            type='rect',
            xref='x',
            yref='paper',
            x0=start_date,
            y0=0,
            x1=end_date,
            y1=1,
            fillcolor=color,
            opacity=0.5,
            layer='below',
            line_width=0,
        )
        return fig

    def format_annotation(
        self,
        fig: go.Figure,
        ax: float = 0,
        ay: float = -40,
        showarrow: bool = True,
        arrowhead: float = 7,
    ) -> go.Figure:
        fig.update_annotations(dict(
                    xref="x",
                    yref="y",
                    showarrow=showarrow,
                    arrowhead=arrowhead,
                    ax=ax,
                    ay=ay,
        ))
        return fig
plotly_util = PlotlyFigure()
plt.style.use('ggplot')
random.seed(42)
HISTORY_COUNTS = 1913
PRED_COUNTS = 28
NUM_ITEMS = 30490

HISTORY_COLUMNS = [f'd_{i + 1}' for i in range(HISTORY_COUNTS)]
calendar = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')
sales = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv')
prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
show_basic_info(calendar)
show_basic_info(sales)
show_basic_info(prices)
fig, axes = plt.subplots(5, 2, figsize=(15, 10))
samples = random.sample(range(len(sales)), 10)

x = pd.to_datetime(calendar.iloc[:HISTORY_COUNTS, :]['date']).values
for ax, sample in zip(axes.flatten(), samples):
    y = sales[HISTORY_COLUMNS].loc[sample, :]
    ax.plot(x, y)
    ax.set_title(sales.loc[sample, 'id'])
    ax.set_xlabel('date')
    ax.set_ylabel('sales')

plt.tight_layout()
plt.show()
fig, axes = plt.subplots(5, 2, figsize=(15, 10))
samples = random.sample(range(len(sales)), 10)

for ax, sample in zip(axes.flatten(), samples):
    item = sales['item_id'].loc[sample]
    store = sales['store_id'].loc[sample]
    ax.plot(
        prices[(prices['item_id']==item) & (prices['store_id']==store)]['wm_yr_wk'],
        prices[(prices['item_id']==item) & (prices['store_id']==store)]['sell_price']
    )
    ax.set_title(f"Price of {sales['id'].loc[sample]}")
    ax.set_xlabel('week number')
    ax.set_ylabel('price')
    
plt.tight_layout()
plt.show()
replacer = ReplaceBeforeFirstSell()
replacer.transform(sales, HISTORY_COLUMNS)
show_basic_info(sales)
x = calendar.iloc[:HISTORY_COUNTS, :]['date'].values
total_sales = sales[HISTORY_COLUMNS].sum()
fig = go.Figure(
    data=[go.Scatter(x=x, y=total_sales.values, name='raw')],
)
fig.add_trace(go.Scatter(x=x, y=total_sales.rolling(7).mean().values, name='1 week MA'))
fig.add_trace(go.Scatter(x=x, y=total_sales.rolling(28).mean().values, name='4 week MA'))
fig.add_trace(go.Scatter(x=x, y=total_sales.rolling(90).mean().values, name='90 days MA'))
fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')
fig.update_layout(
    title_text='All item sales(sum)',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
x = calendar.iloc[:HISTORY_COUNTS, :]['date'].values
total_sales = sales[HISTORY_COLUMNS].mean()
fig = go.Figure(
    data=[go.Scatter(x=x, y=total_sales.values, name='raw')],
)
fig.add_trace(go.Scatter(x=x, y=total_sales.rolling(7).mean().values, name='1 week MA'))
fig.add_trace(go.Scatter(x=x, y=total_sales.rolling(28).mean().values, name='4 week MA'))
fig.add_trace(go.Scatter(x=x, y=total_sales.rolling(90).mean().values, name='90 days MA'))
fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')
fig.update_layout(
    title_text='All item sales(mean)',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
date_map = {i + 1: datetime.datetime.strptime(calendar['date'].min(
), '%Y-%m-%d') + datetime.timedelta(days=i) for i in range(HISTORY_COUNTS)}

sales['first_sold'] = HISTORY_COUNTS - sales[HISTORY_COLUMNS].count(axis=1) + 1
sales['first_sold'] = sales['first_sold'].replace(date_map)

first_sales = sales.groupby('first_sold').size(
).reset_index().rename(columns={0: 'count'})
first_sales['cumulative count'] = first_sales['count'].cumsum()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x=first_sales['first_sold'], y=first_sales['count'], mode='markers', name='First sold count'),
    secondary_y=False,)
fig.add_trace(go.Scatter(
    x=first_sales['first_sold'], y=first_sales['cumulative count'],
    name='Cumulative count', line = dict(color='firebrick', dash='dot')),
    secondary_y=True,)
fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')

fig.update_layout(
    title_text='Daily counts of item sold first time',
    xaxis_title='date',
)
fig.update_yaxes(title_text="Item count", secondary_y=False)
fig.update_yaxes(title_text="Cumulative item count", secondary_y=True)
fig.show()
x = calendar.iloc[:HISTORY_COUNTS, :]['date'].values
unique_sales = (sales[HISTORY_COLUMNS].fillna(0) > 0).sum()
fig = go.Figure(
    data=[go.Scatter(x=x, y=unique_sales.values, name='raw')],
)
fig.add_trace(go.Scatter(x=x, y=unique_sales.rolling(7).mean().values, name='1 week MA'))
fig.add_trace(go.Scatter(x=x, y=unique_sales.rolling(28).mean().values, name='4 week MA'))
fig.add_trace(go.Scatter(x=x, y=unique_sales.rolling(90).mean().values, name='90 days MA'))
fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')
fig.update_layout(
    title_text='Unique Sales',
    xaxis_title='date',
    yaxis_title='unique sales count',
)
fig.show()
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

sales.groupby(['cat_id'])['item_id'].\
    count()[::-1].plot.barh(ax=ax1)

sales[sales['cat_id']=='FOODS'].\
    groupby(['dept_id'])['item_id'].\
    count()[::-1].plot.barh(ax=ax2)

sales[sales['cat_id']=='HOUSEHOLD'].\
    groupby(['dept_id'])['item_id'].\
    count()[::-1].plot.barh(ax=ax3)

sales[sales['cat_id']=='HOBBIES'].\
    groupby(['dept_id'])['item_id'].\
    count()[::-1].plot.barh(ax=ax4)

ax1.set_title('Item counts per category')
ax1.set_xlabel('item_count')
ax2.set_title('FOODS item counts per department')
ax2.set_xlabel('item_count')
ax3.set_title('HOUSEHOLDS item counts per department')
ax3.set_xlabel('item_count')
ax4.set_title('HOBBIES item counts per department')
ax4.set_xlabel('item_count')
fig.tight_layout()
plt.show()
fig = plt.figure(figsize=(12, 6))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

sales.groupby(['cat_id'])[HISTORY_COLUMNS].\
    sum().sum(axis=1)[::-1].plot.barh(ax=ax1)

sales[sales['cat_id']=='FOODS'].\
    groupby(['dept_id'])[HISTORY_COLUMNS].\
    sum().sum(axis=1)[::-1].plot.barh(ax=ax2)

sales[sales['cat_id']=='HOUSEHOLD'].\
    groupby(['dept_id'])[HISTORY_COLUMNS].\
    sum().sum(axis=1)[::-1].plot.barh(ax=ax3)

sales[sales['cat_id']=='HOBBIES'].\
    groupby(['dept_id'])[HISTORY_COLUMNS].\
    sum().sum(axis=1)[::-1].plot.barh(ax=ax4)

ax1.set_title('Item sales per category')
ax1.set_xlabel('item_count')
ax2.set_title('FOODS item counts per department')
ax2.set_xlabel('item_count')
ax3.set_title('HOUSEHOLDS item counts per department')
ax3.set_xlabel('item_count')
ax4.set_title('HOBBIES item counts per department')
ax4.set_xlabel('item_count')
fig.tight_layout()
plt.show()
dept_sales = sales.groupby(['dept_id'])[HISTORY_COLUMNS].mean().T
dept_sales = dept_sales.loc[calendar.loc[:HISTORY_COUNTS-1, :][~(calendar['event_name_1'] == 'Christmas')]['d'].values, :]

fig = go.Figure()
for col in dept_sales:
    fig.add_trace(go.Box(x=[col]*len(dept_sales), y=dept_sales[col], name=col))
    
fig.update_layout(
    title_text='Unit sales box plot per department',
    xaxis_title='dept_id',
    yaxis_title='mean sales per product',
)
fig.show()
x = calendar.iloc[:HISTORY_COUNTS, :]['date'].values
dept_sales = sales.groupby(['dept_id'])[HISTORY_COLUMNS].sum()

fig = go.Figure()
for dept in dept_sales.index:
    y = dept_sales.loc[dept, :].values
    fig.add_trace(go.Scatter(x=x, y=y, name=dept))

fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')
fig.update_layout(
    title_text='Item sales per department',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
fig = go.Figure()
for dept in dept_sales.index:
    y = dept_sales.loc[dept, :].rolling(7).mean().values
    fig.add_trace(go.Scatter(x=x, y=y, name=dept))

fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')
fig.update_layout(
    title_text='Item sales per department(1week MA)',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
dept_sales = sales.groupby(['dept_id'])[HISTORY_COLUMNS].sum().T
dept_sales_weekly = pd.merge(dept_sales, calendar[['d', 'weekday']], 
         left_index=True, right_on=['d']).groupby(['weekday']).sum()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
sns.heatmap(dept_sales_weekly.T.apply(lambda x: x / x.sum(), axis=1)
            [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']],
            cmap='Blues',
            annot=True,
            fmt='.3f',
            linewidths=.5)
ax.set_title('Weekday sales rate of each department', size=14)
plt.show()
df_values = sales[HISTORY_COLUMNS].values
tmp = np.tile(np.arange(1, len(HISTORY_COLUMNS) + 1)[::-1],
              (df_values.shape[0], 1))
tmp_values = ((df_values) > 0) * tmp
last_no = np.min(np.where(tmp_values == 0, 9999, tmp_values), axis=1) - 1
sales['last_sold'] = pd.to_datetime(calendar.loc[HISTORY_COUNTS - last_no - 1, 'date'].values)

last_sales = sales.groupby('last_sold').size(
).reset_index().rename(columns={0: 'count'})
last_sales['cumulative count'] = last_sales['count'].cumsum()
fig = make_subplots(specs=[[{"secondary_y": True}]])
fig.add_trace(go.Scatter(
    x=last_sales['last_sold'], y=last_sales['count'], mode='markers', name='Last sold item count'),
    secondary_y=False,)
fig.add_trace(go.Scatter(
    x=last_sales['last_sold'], y=last_sales['cumulative count'],
    name='Cumulative count', line = dict(color='firebrick', dash='dot')),
    secondary_y=True,)
fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')

fig.update_layout(
    title_text='Daily counts of item sold last time',
    xaxis_title='date',
)
fig.update_yaxes(title_text="Item count", secondary_y=False)
fig.update_yaxes(title_text="Cumulative item count", secondary_y=True)
fig.show()
print(f"95%: {last_sales[(last_sales['cumulative count'] / NUM_ITEMS) <= 0.05]['last_sold'].max()}")
print(f"99%: {last_sales[(last_sales['cumulative count'] / NUM_ITEMS) <= 0.01]['last_sold'].max()}")
store_sales = sales.groupby(['store_id'])[HISTORY_COLUMNS].mean().T
store_sales = store_sales.loc[calendar.loc[:HISTORY_COUNTS-1, :][~(calendar['event_name_1'] == 'Christmas')]['d'].values, :]

fig = go.Figure()
for col in store_sales:
    fig.add_trace(go.Box(x=[col]*len(store_sales), y=store_sales[col], name=col))
    
fig.update_layout(
    title_text='Unit sales box plot per store',
    xaxis_title='store_id',
    yaxis_title='mean sales per product',
)
fig.show()
x = calendar.iloc[:HISTORY_COUNTS, :]['date'].values
store_sales = sales.groupby(['store_id'])[HISTORY_COLUMNS].sum()

fig = go.Figure()
for store in store_sales.index:
    y = store_sales.loc[store, :].values
    fig.add_trace(go.Scatter(x=x, y=y, name=store))

fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')
fig.update_layout(
    title_text='Item sales per store',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
fig = go.Figure()
for store in store_sales.index:
    y = store_sales.loc[store, :].rolling(7).mean().values
    fig.add_trace(go.Scatter(x=x, y=y, name=store))

fig = plotly_util.add_range_selector(fig)
fig = plotly_util.add_shape_region(fig, '2016-04-25', '2016-05-23', 'LightSeaGreen')
fig = plotly_util.add_shape_region(fig, '2016-05-23', '2016-06-19', 'LightPink')
fig.update_layout(
    title_text='Item sales per store (1week MA)',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
store_sales = sales.groupby(['store_id'])[HISTORY_COLUMNS].sum().T
store_sales_weekly = pd.merge(store_sales, calendar[['d', 'weekday']], 
         left_index=True, right_on=['d']).groupby(['weekday']).sum()


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
sns.heatmap(store_sales_weekly.T.apply(lambda x: x / x.sum(), axis=1)
            [['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']],
            cmap='Blues',
            annot=True,
            fmt='.3f',
            linewidths=.5)
ax.set_title('Weekday sales rate of each store', size=14)
plt.show()
def random_plot(item_id=None):
    """
    item_idを引数に渡すことで描画するアイテムを直接指定することも可能
    """
    fig, axes = plt.subplots(5, 2, figsize=(15, 10))
    if item_id is None:
        item = random.sample(list(sales['item_id'].unique()), 1)[0]
    else:
        item = item_id

    x = pd.to_datetime(calendar.iloc[:HISTORY_COUNTS, :]['date']).values
    for ax, store in zip(axes.flatten(), sales['store_id'].unique()):
        y = sales[(sales['item_id']==item) & (sales['store_id']==store)][HISTORY_COLUMNS].values[0]
        ax.plot(x, y)
        ax.set_title(store)
        ax.set_xlabel('date')
        ax.set_ylabel('sales')

    fig.suptitle(f'{item} sales per store', fontsize=16)
    plt.tight_layout()
    plt.show()
random_plot('FOODS_3_090')
random_plot('HOUSEHOLD_1_528')
random_plot('FOODS_2_243')
store_dept_sales = sales.groupby(['store_id', 'dept_id']).sum().sum(axis=1).unstack()
store_dept_sales = store_dept_sales.apply(lambda x: x / x.sum(), axis=1)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111)
sns.heatmap(store_dept_sales,
            cmap='RdPu',
            annot=True,
            fmt='.3f',
            linewidths=.5)
ax.set_title('Department sales rate of each store', size=14)
plt.show()
snap_ca = calendar.iloc[:HISTORY_COUNTS, :][['date', 'weekday', 'snap_CA']].set_index('date')
snap_tx = calendar.iloc[:HISTORY_COUNTS, :][['date', 'weekday', 'snap_TX']].set_index('date')
snap_wi = calendar.iloc[:HISTORY_COUNTS, :][['date', 'weekday', 'snap_WI']].set_index('date')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))
snap_ca['snap_CA'].value_counts().plot.bar(ax=ax1)
snap_tx['snap_TX'].value_counts().plot.bar(ax=ax2)
snap_wi['snap_WI'].value_counts().plot.bar(ax=ax3)

ax1.set_title('SNAP days in CA')
ax2.set_title('SNAP days in TX')
ax3.set_title('SNAP days in WI')
ax1.set_xticklabels(['No SNAP', 'SNAP'])
ax2.set_xticklabels(['No SNAP', 'SNAP'])
ax3.set_xticklabels(['No SNAP', 'SNAP'])
ax1.set_ylabel('Day counts')
ax2.set_ylabel('Day counts')
ax3.set_ylabel('Day counts')
plt.tight_layout()

fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111)

calendar.iloc[:HISTORY_COUNTS, :][['date', 'weekday', 'snap_CA', 'snap_TX', 'snap_WI']].\
    set_index('date').groupby('weekday').sum().plot.barh(ax=ax)
ax.set_title('SNAP day counts per weekday')

plt.show()
cax1 = snap_ca[snap_ca['snap_CA']==1].reset_index()['date'].values
cay1 = np.extract((snap_ca['snap_CA']==1).values,
                sales[sales['state_id']=='CA'][HISTORY_COLUMNS].sum().values)

cax2 = snap_ca[snap_ca['snap_CA']==0].reset_index()['date'].values
cay2 = np.extract((snap_ca['snap_CA']==0).values,
                sales[sales['state_id']=='CA'][HISTORY_COLUMNS].sum().values)

txx1 = snap_tx[snap_tx['snap_TX']==1].reset_index()['date'].values
txy1 = np.extract((snap_tx['snap_TX']==1).values,
                sales[sales['state_id']=='TX'][HISTORY_COLUMNS].sum().values)

txx2 = snap_tx[snap_tx['snap_TX']==0].reset_index()['date'].values
txy2 = np.extract((snap_tx['snap_TX']==0).values,
                sales[sales['state_id']=='TX'][HISTORY_COLUMNS].sum().values)

wix1 = snap_wi[snap_wi['snap_WI']==1].reset_index()['date'].values
wiy1 = np.extract((snap_wi['snap_WI']==1).values,
                sales[sales['state_id']=='WI'][HISTORY_COLUMNS].sum().values)

wix2 = snap_wi[snap_wi['snap_WI']==0].reset_index()['date'].values
wiy2 = np.extract((snap_wi['snap_WI']==0).values,
                sales[sales['state_id']=='WI'][HISTORY_COLUMNS].sum().values)


fig = make_subplots(rows=3, cols=1, subplot_titles=('Item Sales in CA', 'Item Sales in TX', 'Item Sales in WI'))
fig.add_trace(go.Scatter(x=cax1, y=cay1, name='SNAP(CA)'), row=1, col=1)
fig.add_trace(go.Scatter(x=cax2, y=cay2, name='No SNAP(CA)'), row=1, col=1)
fig.add_trace(go.Scatter(x=txx1, y=txy1, name='SNAP(TX)'), row=2, col=1)
fig.add_trace(go.Scatter(x=txx2, y=txy2, name='No SNAP(TX)'), row=2, col=1)
fig.add_trace(go.Scatter(x=wix1, y=wiy1, name='SNAP(WI)'), row=3, col=1)
fig.add_trace(go.Scatter(x=wix2, y=wiy2, name='No SNAP(WI)'), row=3, col=1)

fig = plotly_util.add_range_selector(fig)
fig.update_layout(
    height=1000,
    title_text='Item sales by SNAP and No SNAP',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
fig = make_subplots(rows=1, cols=3, shared_yaxes=True,
                    subplot_titles=(f'CA ({cay1.mean() / cay2.mean():.3f})',
                                    f'TX ({txy1.mean() / txy2.mean():.3f})',
                                    f'WI ({wiy1.mean() / wiy2.mean():.3f})'))

fig.add_trace(go.Bar(x=['SNAP', 'No SNAP'], y=[cay1.mean(), cay2.mean()], name='CA'), row=1, col=1)
fig.add_trace(go.Bar(x=['SNAP', 'No SNAP'], y=[txy1.mean(), txy2.mean()], name='TX'), row=1, col=2)
fig.add_trace(go.Bar(x=['SNAP', 'No SNAP'], y=[wiy1.mean(), wiy2.mean()], name='WI'), row=1, col=3)

fig.update_layout(
    height=400,
    title_text='Item average daily sales by SNAP and No SNAP',
    xaxis_title='date',
    yaxis_title='sales',
)
fig.show()
fig = make_subplots(rows=6, cols=2)
total_sales = sales[HISTORY_COLUMNS].sum()

for month in range(1, 13):
    x = calendar[(calendar['year'] == 2015) & (calendar['month'] == month)]['date'].values
    events = calendar[(calendar['event_name_1'].notnull()) &
                      (calendar['year'] == 2015) &
                      (calendar['month'] == month)]
    fig = fig.add_trace(go.Scatter(
        x=x,
        y=total_sales.loc[
            calendar[(calendar['year'] == 2015) & (calendar['month'] == month)]['d']
            ].values,
        name=f'month: {month}',
        ),
        row=(month - 1) // 2 + 1,
        col=(month - 1) % 2 + 1,
        )
    for idx in events.index:
        fig.add_annotation(x=events.loc[idx, 'date'],
                           y=total_sales.loc[events.loc[idx, 'd']],
                           text=events.loc[idx, 'event_name_1'],
                           row=(month - 1) // 2 + 1,
                           col=(month - 1) % 2 + 1)
# fig = plotly_util.format_annotation(fig)
fig.update_layout(
    height=1200,
    title_text='Item Sales in 2015 with events',
)
fig.show()