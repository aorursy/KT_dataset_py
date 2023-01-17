# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import gc

import matplotlib.pyplot as plt

import seaborn as sns

from scipy.interpolate import make_interp_spline

from tqdm._tqdm_notebook import tqdm_notebook

import random

%matplotlib inline

sns.set_style("darkgrid")

import plotly.graph_objs as go

import plotly.express as px
df_train = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_validation.csv')

df_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

df_calender = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')
# for reducing memory usage

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)    

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df
%%time

df_train = reduce_mem_usage(df_train)

df_prices = reduce_mem_usage(df_prices)

df_calender = reduce_mem_usage(df_calender)
df_train.head()
df_prices.head()
df_calender.head()
df_calender.info()
df_calender['date'] = pd.to_datetime(df_calender['date']).dt.date
df_calender['date'].min(), df_calender['date'].max()
# we will keep only data from jan 2015

# find specific value of d for that date

day_value = df_calender[df_calender['date']==pd.Timestamp(year=2015, month=1, day=1)]['d'].values[0]

day_value
# drop from train data

days_to_drop = [f'd_{i}' for i in range(1, int(day_value.split('_')[-1]))]

df_train.drop(days_to_drop, axis = 1, inplace = True)
df_train.head()
df_train = pd.melt(df_train, \

                   id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], \

                   var_name = 'day', value_name = 'demand')

#df_train.sort_values(by = 'id', ascending=True, inplace=True)

df_train.head()
df_train = reduce_mem_usage(df_train)
df_train.head()
#shape of train data

df_train.shape
# find unique values

for col in df_train.columns:

    print (f'{col}:{df_train[col].nunique()}')
#find missing values

pd.isnull(df_train).any().any()
# remove d from day columns

df_train['day'] = df_train['day'].apply(lambda x : int(x.split('_')[-1]))
df_prices.head()
# remove d from day columns for joining with train data

df_calender['d'] = df_calender['d'].apply(lambda x : int(x.split('_')[-1]))

df_calender.head()
df_calender['date'] = pd.to_datetime(df_calender['date']).dt.date

df_calender['date'].describe()
# keep only dates from 2015

df_calender = df_calender[df_calender['year'].isin([2015, 2016])]

df_calender.shape
df_calender.describe()
for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:

    print (f'{col} : {df_calender[col].unique()}')
# encode event names for reducing memory

map_event_name_1 = {'NewYear': 1,

 'OrthodoxChristmas': 2,

 'MartinLutherKingDay': 3,

 'SuperBowl': 4,

 'ValentinesDay': 5,

 'PresidentsDay': 6,

 'LentStart': 7,

 'LentWeek2': 8,

 'Purim End': 9,

 'StPatricksDay': 10,

 'Easter': 11,

 'Pesach End': 12,

 'Cinco De Mayo': 13,

 "Mother's day": 14,

 'MemorialDay': 15,

 'NBAFinalsStart': 16,

 'NBAFinalsEnd': 17,

 'Ramadan starts': 18,

 'IndependenceDay': 19,

 'Eid al-Fitr': 20,

 'LaborDay': 21,

 'EidAlAdha': 22,

 'ColumbusDay': 23,

 'Halloween': 24,

 'VeteransDay': 25,

 'Thanksgiving': 26,

 'Chanukah End': 27,

 'Christmas': 28,

 'OrthodoxEaster': 29,

 "Father's day": 30}



map_event_name_2 = {'OrthodoxEaster': 1, "Father's day": 2, 'Easter':3, 'Cinco De Mayo':4}



map_event_type_1 = {'National': 1, 'Religious': 2, 'Sporting': 3, 'Cultural': 4}



map_event_type_2 = {'Religious': 1, 'Cultural': 2}



map_weekday = {'Monday': 1,

 'Tuesday': 2,

 'Wednesday': 3,

 'Thursday': 4,

 'Friday': 5,

 'Saturday': 6,

 'Sunday': 7}
df_calender['event_name_1'] = df_calender['event_name_1'].apply(lambda x : map_event_name_1[x] if x == x else 0)

df_calender['event_name_2'] = df_calender['event_name_2'].apply(lambda x : map_event_name_2[x] if x == x else 0)

df_calender['event_type_1'] = df_calender['event_type_1'].apply(lambda x : map_event_type_1[x] if x == x else 0)

df_calender['event_type_2'] = df_calender['event_type_2'].apply(lambda x : map_event_type_2[x] if x == x else 0)

df_calender['weekday'] = df_calender['weekday'].apply(lambda x : map_weekday[x])
for col in ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']:

    print (f'{col} : {df_calender[col].unique()}')
df_train = pd.merge(df_train, df_calender[['wm_yr_wk', 'd', 'weekday', 'event_name_1', 'event_type_1', 'event_name_2', \

                                           'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI']], \

                    left_on = 'day', \

                    right_on = 'd', \

                    how = 'left')

df_train.drop(['d'], axis = 1, inplace = True)

df_train.head()
df_train.shape
df_train.sort_values(by = ['id', 'day'], inplace = True)

df_train.reset_index(drop = True, inplace = True)

df_train.head()
%%time

# clean snap columns 

# snap values are corresponding to particular states 

def clean_snap_columns(x):

    state = x['state_id']

    if state == 'CA':

        snap = x['snap_CA']

    elif state == 'TX':

        snap = x['snap_TX']

    else:

        snap = x['snap_WI']

    return snap

df_train['snap'] = df_train.apply(lambda x : clean_snap_columns(x), axis = 1)

df_train.drop(['snap_CA', 'snap_TX', 'snap_WI'], axis = 1, inplace = True)
df_train.head()
df_train.shape
del df_calender
#is there any negative demand 

df_train[df_train['demand']<0]
# do some depts have more sales

def plot_day_values(group_col, type_of_plot):

    grouped = df_train.groupby([group_col, 'day'])['demand'].mean().reset_index()

    if type_of_plot == 'line':

        fig = px.line(grouped, x = 'day', y = 'demand', color=group_col)

        fig.show()

    if type_of_plot == 'bar':

        fig = px.bar(grouped, x = group_col, y = 'demand')

        fig.show()

    return
# which department has highest average sales

plot_day_values('dept_id', 'line')
plot_day_values('dept_id', 'bar')
# which store has highest average sales

plot_day_values('store_id', 'line')
plot_day_values('store_id', 'bar')
# which state has highest average sales

plot_day_values('state_id', 'line')
plot_day_values('state_id', 'bar')
df_prices.head()
# merge prices in train data

df_train = pd.merge(df_train, df_prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')
df_train = reduce_mem_usage(df_train)
df_train.head()
def plot_wd_values(group_col):

    grouped = df_train.groupby([group_col, 'weekday'])['demand'].mean().reset_index()

    fig = go.Figure()

    for col in grouped[group_col].unique():

        grouped_col = grouped[grouped[group_col]==col]

        trace = go.Scatter(x = grouped_col['weekday'], y = grouped_col['demand'], name = col)

        fig.add_trace(trace)

    fig.show()

    return
# dept sales on weekdays

plot_wd_values('dept_id')
# store sales on weekdays

plot_wd_values('store_id')
# state sales on weekdays

plot_wd_values('state_id')
# snap

df_train['snap'].value_counts(normalize = True)
def plot_snap_values(group_col):

    plt.figure(figsize = (24, 4))

    grouped = df_train.groupby([group_col, 'snap'])['demand'].mean().reset_index()

    sns.barplot(x = group_col, y = 'demand', hue = 'snap', data = grouped, hue_order = [0, 1])

    return
# dept sales for snap days

plot_snap_values('dept_id')
# store sales for snap days

plot_snap_values('store_id')
# state sales for snap days

plot_snap_values('state_id')
gc.collect()
for col in ['event_type_1', 'event_type_2']:

    print (df_train[col].value_counts())
def plot_event_type_1_values(group_col):

    plt.figure(figsize = (24, 4))

    grouped = df_train.groupby([group_col, 'event_type_1'])['demand'].mean().reset_index()

    sns.barplot(x = group_col, y = 'demand', hue = 'event_type_1', data = grouped, hue_order = [0, 1, 2, 3, 4])

    return
# dept sales for event type 1

plot_event_type_1_values('dept_id')
# store sales for event type 1

plot_event_type_1_values('store_id')
# state sales for event type 1

plot_event_type_1_values('state_id')
df_train.head()
del df_prices
gc.collect()
# number of unique ids

df_train['id'].nunique()
# some ids do not have demand. 

# which ids are those

grouped = df_train.groupby(['id'])['demand'].sum().reset_index()

grouped[grouped['demand']==0]['id'].unique()
# number of ids having no demand 

grouped[grouped['demand']==0]['id'].nunique()
# plot demands of random 5 item ids

random.seed(12)

ids_sold = [train_id for train_id in df_train['id'].unique().tolist() if train_id not in grouped[grouped['demand']==0]['id'].tolist()]

random_ids = random.sample(ids_sold, 5)

def plot_demand(random_ids):

    fig = go.Figure()

    for i, random_id in enumerate(random_ids):

        temp_df = df_train[df_train['id']==random_id]

        trace = go.Scatter(x = temp_df['day'], y = temp_df['demand'], mode = "lines+markers", name = random_id)

        fig.add_trace(trace)

    fig.show()

    return



plot_demand(random_ids)
# plot demands of random 1 item id which had no demand

random.seed(12)

#ids_sold = [train_id for train_id in df_train['id'].unique().tolist() if train_id in grouped[grouped['demand']==0]['id'].tolist()]

random_ids = random.sample(grouped[grouped['demand']==0]['id'].tolist(), 1)

def plot_demand(random_ids):

    fig = go.Figure()

    for i, random_id in enumerate(random_ids):

        temp_df = df_train[df_train['id']==random_id]

        trace = go.Scatter(x = temp_df['day'], y = temp_df['demand'], mode = "lines+markers", name = random_id)

        fig.add_trace(trace)

    fig.show()

    return



plot_demand(random_ids)
# how many ids have constant prices and different prices

grouped = df_train.groupby('id')['sell_price'].nunique().reset_index()

ids_price_no_change = grouped[grouped['sell_price']==1]

ids_price_with_change = grouped[grouped['sell_price']>1]

#grouped = df_train.groupby('id')['demand'].sum().reset_index()

#ids_no_demand = grouped[grouped['demand']==0]

#df_train['id'].nunique(), ids_price_no_change.shape[0], ids_no_demand.shape[0]

ids_price_no_change['id'].nunique(), ids_price_with_change['id'].nunique()
# how many days are there

df_train['day'].nunique()
# distribution of demand and no demand

demand_explore = df_train[['id', 'day', 'demand']]

demand_explore['is_demand'] = demand_explore['demand'].apply(lambda x : int(x>=1))

demand_explore = pd.crosstab(demand_explore['id'], demand_explore['is_demand'], values = demand_explore['day'], aggfunc = 'count').reset_index().fillna(0)

demand_explore.head()
# which item id had lowest no demand days

demand_explore.sort_values(by = demand_explore.columns[2], ascending=False).head()
# how many days are there on average where there is no demand

demand_explore.rename(columns = {0:'no_demand', 1:'demand'}, inplace = True)

demand_explore[['no_demand', 'demand']].mean()
# lets explore a bit more regarding catergory, dept, store, state

demand_explore = pd.merge(demand_explore, df_train[['id', 'dept_id', 'cat_id', 'store_id', 'state_id']], how = 'left', on = 'id')

demand_explore.head()
# do some depts have more sales

def plot_demand_days(group_col):

    grouped = demand_explore.groupby([group_col]).agg({'demand':np.mean}).reset_index()

    fig = px.bar(grouped, x = group_col, y = 'demand', labels={'demand':'days of demand'})

    fig.show()

    return
plot_demand_days('dept_id')
plot_demand_days('cat_id')
plot_demand_days('state_id')
plot_demand_days('store_id')