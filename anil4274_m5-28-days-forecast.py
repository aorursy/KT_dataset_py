import os

import pandas as pd

import numpy as np

import plotly_express as px

import plotly.graph_objects as go

from plotly.subplots import make_subplots

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')
sales = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')
sales_bmr = np.round(sales.memory_usage().sum()/(1024*1024),1)

calendar_bmr = np.round(calendar.memory_usage().sum()/(1024*1024),1)

prices_bmr = np.round(prices.memory_usage().sum()/(1024*1024),1)
def reduceMemory(df):

    columns = df.dtypes.index.tolist()

    datypes = df.dtypes.values.tolist()

    for i,j in enumerate(datypes):

        if 'int' in str(j):

            if df[columns[i]].min() > np.iinfo(np.int8).min and df[columns[i]].max() < np.iinfo(np.int8).max:

                df[columns[i]] = df[columns[i]].astype(np.int8)

            elif df[columns[i]].min() > np.iinfo(np.int16).min and df[columns[i]].max() < np.iinfo(np.int16).max:

                df[columns[i]] = df[columns[i]].astype(np.int16)

            elif df[columns[i]].min() > np.iinfo(np.int32).min and df[columns[i]].max() < np.iinfo(np.int32).max:

                df[columns[i]] = df[columns[i]].astype(np.int32)

            else:

                df[columns[i]] = df[columns[i]].astype(np.int64)

        elif 'float' in str(j):

            if df[columns[i]].min() > np.finfo(np.float16).min and df[columns[i]].max() < np.finfo(np.float16).max:

                df[columns[i]] = df[columns[i]].astype(np.float16)

            elif df[columns[i]].min() > np.finfo(np.float32).min and df[columns[i]].max() < np.finfo(np.float32).max:

                df[columns[i]] = df[columns[i]].astype(np.float32)

            else:

                df[columns[i]] = df[columns[i]].astype(np.float64)

        elif j == np.object:

            if columns[i] == 'date':

                df[columns[i]] = pd.to_datetime(df[columns[i]], format='%Y-%m-%d')

            else:

                df[columns[i]] = df[columns[i]].astype('category')

    return df  



sales = reduceMemory(sales)

prices = reduceMemory(prices)

calendar = reduceMemory(calendar)
sales_amr = np.round(sales.memory_usage().sum()/(1024*1024),1)

calendar_amr = np.round(calendar.memory_usage().sum()/(1024*1024),1)

prices_amr = np.round(prices.memory_usage().sum()/(1024*1024),1)
d = {'DataFrame':['sales','calendar','prices'],

       'Before Memory Reduction':[sales_bmr,calendar_bmr,prices_bmr],

       'After Memory Reduction':[sales_amr,calendar_amr,prices_amr]}



memory = pd.DataFrame(d)

memory = pd.melt(memory, id_vars='DataFrame', var_name='Status', value_name='Memory (MB)')

memory.sort_values('Memory (MB)',inplace=True)

fig = px.bar(memory, x='DataFrame', y='Memory (MB)', color='Status', barmode='group', text='Memory (MB)')

fig.update_traces(texttemplate='%{text} MB', textposition='outside')

fig.update_layout(template='seaborn', title='Memory Saving')

fig.show()
df = pd.melt(sales, id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name='d', value_name='sold').dropna()
df = pd.merge(df, calendar, on='d', how='left')

df = pd.merge(df, prices, on=['store_id','item_id','wm_yr_wk'], how='left') 
sales.head()
group = sales.groupby(['state_id','store_id','cat_id','dept_id'],as_index=False)['item_id'].count().dropna()

group['USA'] = 'United States of America'

group.rename(columns={'state_id':'State','store_id':'Store','cat_id':'Category','dept_id':'Department','item_id':'Count'},inplace=True)

fig = px.treemap(group, path=['USA', 'State', 'Store', 'Category', 'Department'], values='Count',

                  color='Count',

                  color_continuous_scale= px.colors.sequential.Sunset,

                  title='Walmart: Distribution of items')

fig.update_layout(template='seaborn')

fig.show()
group_price_store = df.groupby(['state_id','store_id','item_id'],as_index=False)['sell_price'].mean().dropna()

fig = px.violin(group_price_store, x='store_id', color='state_id', y='sell_price',box=True, hover_name='item_id')

fig.update_xaxes(title_text='Store')

fig.update_yaxes(title_text='Selling Price($)')

fig.update_layout(template='seaborn',title='Distribution of Items prices wrt Stores',legend_title_text='State')

fig.show()
group_price_cat = df.groupby(['store_id','cat_id','item_id'],as_index=False)['sell_price'].mean().dropna()

fig = px.violin(group_price_cat, x='store_id', color='cat_id', y='sell_price',box=True, hover_name='item_id')

fig.update_xaxes(title_text='Store')

fig.update_yaxes(title_text='Selling Price($)')

fig.update_layout(template='seaborn',title='Distribution of Items prices wrt Stores across Categories',

                 legend_title_text='Category')

fig.show()
group = df.groupby(['year','date','state_id','store_id'], as_index=False)['sold'].sum().dropna()

fig = px.violin(group, x='store_id', color='state_id', y='sold',box=True)

fig.update_xaxes(title_text='Store')

fig.update_yaxes(title_text='Total items sold')

fig.update_layout(template='seaborn',title='Distribution of Items sold wrt Stores',legend_title_text='State')

fig.show()
fig = go.Figure()

title = 'Items sold over time'

years = group.year.unique().tolist()

buttons = []

y=3

for state in group.state_id.unique().tolist():

    group_state = group[group['state_id']==state]

    for store in group_state.store_id.unique().tolist():

        group_state_store = group_state[group_state['store_id']==store]

        fig.add_trace(go.Scatter(name=store, x=group_state_store['date'], y=group_state_store['sold'], showlegend=True, 

                                   yaxis='y'+str(y) if y!=1 else 'y'))

    y-=1



fig.update_layout(

        xaxis=dict(

        #autorange=True,

        range = ['2011-01-29','2016-05-22'],

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(count=2,

                     label="2y",

                     step="year",

                     stepmode="backward"),

                dict(count=3,

                     label="3y",

                     step="year",

                     stepmode="backward"),

                dict(count=4,

                     label="4y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            autorange=True,

        ),

        type="date"

    ),

    yaxis=dict(

        anchor="x",

        autorange=True,

        domain=[0, 0.33],

        mirror=True,

        showline=True,

        side="left",

        tickfont={"size":10},

        tickmode="auto",

        ticks="",

        title='WI',

        titlefont={"size":20},

        type="linear",

        zeroline=False

    ),

    yaxis2=dict(

        anchor="x",

        autorange=True,

        domain=[0.33, 0.66],

        mirror=True,

        showline=True,

        side="left",

        tickfont={"size":10},

        tickmode="auto",

        ticks="",

        title = 'TX',

        titlefont={"size":20},

        type="linear",

        zeroline=False

    ),

    yaxis3=dict(

        anchor="x",

        autorange=True,

        domain=[0.66, 1],

        mirror=True,

        showline=True,

        side="left",

        tickfont={"size":10},

        tickmode="auto",

        ticks='',

        title="CA",

        titlefont={"size":20},

        type="linear",

        zeroline=False

    )

    )

fig.update_layout(template='seaborn', title=title)

fig.show()
df['revenue'] = df['sold']*df['sell_price'].astype(np.float32)
def introduce_nulls(df):

    idx = pd.date_range(df.date.dt.date.min(), df.date.dt.date.max())

    df = df.set_index('date')

    df = df.reindex(idx)

    df.reset_index(inplace=True)

    df.rename(columns={'index':'date'},inplace=True)

    return df



def plot_metric(df,state,store,metric):

    store_sales = df[(df['state_id']==state)&(df['store_id']==store)&(df['date']<='2016-05-22')]

    food_sales = store_sales[store_sales['cat_id']=='FOODS']

    store_sales = store_sales.groupby(['date','snap_'+state],as_index=False)['sold','revenue'].sum()

    snap_sales = store_sales[store_sales['snap_'+state]==1]

    non_snap_sales = store_sales[store_sales['snap_'+state]==0]

    food_sales = food_sales.groupby(['date','snap_'+state],as_index=False)['sold','revenue'].sum()

    snap_foods = food_sales[food_sales['snap_'+state]==1]

    non_snap_foods = food_sales[food_sales['snap_'+state]==0]

    non_snap_sales = introduce_nulls(non_snap_sales)

    snap_sales = introduce_nulls(snap_sales)

    non_snap_foods = introduce_nulls(non_snap_foods)

    snap_foods = introduce_nulls(snap_foods)

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=non_snap_sales['date'],y=non_snap_sales[metric],

                           name='Total '+metric+'(Non-SNAP)'))

    fig.add_trace(go.Scatter(x=snap_sales['date'],y=snap_sales[metric],

                           name='Total '+metric+'(SNAP)'))

    fig.add_trace(go.Scatter(x=non_snap_foods['date'],y=non_snap_foods[metric],

                           name='Food '+metric+'(Non-SNAP)'))

    fig.add_trace(go.Scatter(x=snap_foods['date'],y=snap_foods[metric],

                           name='Food '+metric+'(SNAP)'))

    fig.update_yaxes(title_text='Total items sold' if metric=='sold' else 'Total revenue($)')

    fig.update_layout(template='seaborn',title=store)

    fig.update_layout(

        xaxis=dict(

        #autorange=True,

        range = ['2011-01-29','2016-05-22'],

        rangeselector=dict(

            buttons=list([

                dict(count=1,

                     label="1m",

                     step="month",

                     stepmode="backward"),

                dict(count=6,

                     label="6m",

                     step="month",

                     stepmode="backward"),

                dict(count=1,

                     label="YTD",

                     step="year",

                     stepmode="todate"),

                dict(count=1,

                     label="1y",

                     step="year",

                     stepmode="backward"),

                dict(count=2,

                     label="2y",

                     step="year",

                     stepmode="backward"),

                dict(count=3,

                     label="3y",

                     step="year",

                     stepmode="backward"),

                dict(count=4,

                     label="4y",

                     step="year",

                     stepmode="backward"),

                dict(step="all")

            ])

        ),

        rangeslider=dict(

            autorange=True,

        ),

        type="date"

    ))

    return fig
cal_data = group.copy()

cal_data = cal_data[cal_data.date <= '22-05-2016']

cal_data['week'] = cal_data.date.dt.weekofyear

cal_data['day_name'] = cal_data.date.dt.day_name()
def calmap(cal_data, state, store, scale):

    cal_data = cal_data[(cal_data['state_id']==state)&(cal_data['store_id']==store)]

    years = cal_data.year.unique().tolist()

    fig = make_subplots(rows=len(years),cols=1,shared_xaxes=True,vertical_spacing=0.005)

    r=1

    for year in years:

        data = cal_data[cal_data['year']==year]

        data = introduce_nulls(data)

        fig.add_trace(go.Heatmap(

            z=data.sold,

            x=data.week,

            y=data.day_name,

            hovertext=data.date.dt.date,

            coloraxis = "coloraxis",name=year,

        ),r,1)

        fig.update_yaxes(title_text=year,tickfont=dict(size=5),row = r,col = 1)

        r+=1

    fig.update_xaxes(range=[1,53],tickfont=dict(size=10), nticks=53)

    fig.update_layout(coloraxis = {'colorscale':scale})

    fig.update_layout(template='seaborn', title=store)

    return fig
fig = plot_metric(df,'CA','CA_1','sold')

fig.show()
fig = plot_metric(df,'CA','CA_1','revenue')

fig.show()
fig = calmap(cal_data, 'CA', 'CA_1', 'magma')

fig.show()
fig = plot_metric(df,'CA','CA_2','sold')

fig.show()
fig = plot_metric(df,'CA','CA_2','revenue')

fig.show()
fig = calmap(cal_data, 'CA', 'CA_2', 'magma')

fig.show()
fig = plot_metric(df,'CA','CA_3','sold')

fig.show()
fig = plot_metric(df,'CA','CA_3','revenue')

fig.show()
fig = calmap(cal_data, 'CA', 'CA_3', 'magma')

fig.show()
fig = plot_metric(df,'CA','CA_4','sold')

fig.show()
fig = plot_metric(df,'CA','CA_4','revenue')

fig.show()
fig = calmap(cal_data, 'CA', 'CA_4', 'magma')

fig.show()
fig = plot_metric(df,'TX','TX_1','sold')

fig.show()
fig = plot_metric(df,'TX','TX_1','revenue')

fig.show()
fig = calmap(cal_data, 'TX', 'TX_1', 'viridis')

fig.show()
fig = plot_metric(df,'TX','TX_2','sold')

fig.show()
fig = plot_metric(df,'TX','TX_2','revenue')

fig.show()
fig = calmap(cal_data, 'TX', 'TX_2', 'viridis')

fig.show()
fig = plot_metric(df,'TX','TX_3','sold')

fig.show()
fig = plot_metric(df,'TX','TX_3','revenue')

fig.show()
fig = calmap(cal_data, 'TX', 'TX_3', 'viridis')

fig.show()
fig = plot_metric(df,'WI','WI_1','sold')

fig.show()
fig = plot_metric(df,'WI','WI_1','revenue')

fig.show()
fig = calmap(cal_data, 'WI', 'WI_1', 'twilight')

fig.show()
fig = plot_metric(df,'WI','WI_2','sold')

fig.show()
fig = plot_metric(df,'WI','WI_2','revenue')

fig.show()
fig = calmap(cal_data, 'WI', 'WI_2', 'twilight')

fig.show()
fig = plot_metric(df, 'WI', 'WI_3', 'sold')

fig.show()
fig = plot_metric(df, 'WI', 'WI_3', 'revenue')

fig.show()
fig = calmap(cal_data, 'WI', 'WI_3', 'twilight')

fig.show()