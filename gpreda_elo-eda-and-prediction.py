import os

import gc

import sys

import random

import logging

import datetime

import numpy as np

import pandas as pd

import seaborn as sns

import lightgbm as lgb

from plotly import tools

from pathlib import Path

import plotly.graph_objs as go

import matplotlib.pyplot as plt

import plotly.figure_factory as ff

from sklearn.metrics import mean_squared_error

from sklearn.model_selection import StratifiedKFold

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
IS_LOCAL = False

if(IS_LOCAL):

    PATH="../input/elo/"

else:

    PATH="../input/"

os.listdir(PATH)
train_df=pd.read_csv(PATH+'train.csv')

test_df=pd.read_csv(PATH+'test.csv')

historical_trans_df=pd.read_csv(PATH+'historical_transactions.csv')

new_merchant_trans_df=pd.read_csv(PATH+'new_merchant_transactions.csv')

merchant_df=pd.read_csv(PATH+'merchants.csv')
print("Train: rows:{} cols:{}".format(train_df.shape[0], train_df.shape[1]))

print("Test:  rows:{} cols:{}".format(test_df.shape[0], test_df.shape[1]))

print("Historical trans: rows:{} cols:{}".format(historical_trans_df.shape[0], historical_trans_df.shape[1]))

print("New merchant trans:  rows:{} cols:{}".format(new_merchant_trans_df.shape[0], new_merchant_trans_df.shape[1]))

print("Merchants: rows:{} cols:{}".format(merchant_df.shape[0], merchant_df.shape[1]))
train_df.sample(3).head()
test_df.sample(3).head()
historical_trans_df.sample(3).head()
new_merchant_trans_df.sample(3).head()
merchant_df.sample(3).head()
def missing_data(data):

    total = data.isnull().sum().sort_values(ascending = False)

    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
missing_data(train_df)
missing_data(test_df)
missing_data(historical_trans_df)
missing_data(new_merchant_trans_df)
missing_data(merchant_df)
def get_categories(data, val):

    tmp = data[val].value_counts()

    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()
def get_target_categories(data, val):

    tmp = data.groupby('target')[val].value_counts()

    return pd.DataFrame(data={'Number': tmp.values}, index=tmp.index).reset_index()
def draw_trace_bar(data_df,color='Blue'):

    trace = go.Bar(

            x = data_df['index'],

            y = data_df['Number'],

            marker=dict(color=color),

            text=data_df['index']

        )

    return trace



def draw_trace_histogram(data_df,target,color='Blue'):

    trace = go.Histogram(

            y = data_df[target],

            marker=dict(color=color)

        )

    return trace
def plot_bar(data_df, title, xlab, ylab,color='Blue'):

    trace = draw_trace_bar(data_df, color)

    data = [trace]

    layout = dict(title = title,

              xaxis = dict(title = xlab, showticklabels=True, tickangle=0,

                          tickfont=dict(

                            size=10,

                            color='black'),), 

              yaxis = dict(title = ylab),

              hovermode = 'closest'

             )

    fig = dict(data = data, layout = layout)

    iplot(fig, filename='draw_trace')
def plot_two_bar(data_df1, data_df2, title1, title2, xlab, ylab):

    trace1 = draw_trace_bar(data_df1, color='Blue')

    trace2 = draw_trace_bar(data_df2, color='Lightblue')

    

    fig = tools.make_subplots(rows=1,cols=2, subplot_titles=(title1,title2))

    fig.append_trace(trace1,1,1)

    fig.append_trace(trace2,1,2)

    

    fig['layout']['xaxis'].update(title = xlab)

    fig['layout']['xaxis2'].update(title = xlab)

    fig['layout']['yaxis'].update(title = ylab)

    fig['layout']['yaxis2'].update(title = ylab)

    fig['layout'].update(showlegend=False)

    

    iplot(fig, filename='draw_trace')
def plot_target_distribution(var):

    hist_data = []

    varall = list(train_df.groupby([var])[var].nunique().index)

    for i, varcrt in enumerate(varall):

        classcrt = train_df[train_df[var] == varcrt]['target']

        hist_data.append(classcrt)

    fig = ff.create_distplot(hist_data, varall, show_hist=False, show_rug=False)

    fig['layout'].update(title='Target variable density plot group by {}'.format(var), xaxis=dict(title='Target'))

    iplot(fig, filename='dist_only')
plot_two_bar(get_categories(train_df,'feature_1'), get_categories(test_df,'feature_1'), 

             'Train data', 'Test data',

             'Feature 1', 'Number of records')
plot_target_distribution('feature_1')
plot_two_bar(get_categories(train_df,'feature_2'), get_categories(test_df,'feature_2'), 

             'Train data', 'Test data',

             'Feature 2', 'Number of records')
plot_target_distribution('feature_2')
plot_two_bar(get_categories(train_df,'feature_3'), get_categories(test_df,'feature_3'), 

             'Train data', 'Test data',

             'Feature 3', 'Number of records')
plot_target_distribution('feature_3')
plot_two_bar(get_categories(train_df,'first_active_month'), get_categories(test_df,'first_active_month'), 

             'Train data', 'Test data',

             'First active month', 'Number of records')
plot_bar(get_categories(historical_trans_df,'category_1'), 

             'Category 1 distribution', 'Category 1', 'Number of records')
plot_bar(get_categories(historical_trans_df,'category_2'), 

             'Category 2 distribution', 'Category 2', 'Number of records','red')
plot_bar(get_categories(historical_trans_df,'category_3'), 

             'Category 3 distribution', 'Category 3', 'Number of records','magenta')
plot_bar(get_categories(historical_trans_df,'city_id'), 

             'City ID distribution', 'City ID', 'Number of records','lightblue')
plot_bar(get_categories(historical_trans_df,'merchant_category_id'), 

             'Merchant Cateogory ID distribution', 'Merchant Category ID', 'Number of records','lightgreen')
plot_bar(get_categories(historical_trans_df,'state_id'), 

             'State ID distribution', 'State ID', 'Number of records','brown')
plot_bar(get_categories(historical_trans_df,'subsector_id'), 

             'Subsector ID distribution', 'Subsector ID', 'Number of records','orange')
historical_trans_df['purchase_date'] = pd.to_datetime(historical_trans_df['purchase_date'])

historical_trans_df['month'] = historical_trans_df['purchase_date'].dt.month

historical_trans_df['dayofweek'] = historical_trans_df['purchase_date'].dt.dayofweek

historical_trans_df['weekofyear'] = historical_trans_df['purchase_date'].dt.weekofyear
def plot_scatter_data(data, xtitle, ytitle, title, color='blue'):

    trace = go.Scatter(

        x = data.index,

        y = data.values,

        name=ytitle,

        marker=dict(

            color=color,

        ),

        mode='lines+markers'

    )

    data = [trace]

    layout = dict(title = title,

              xaxis = dict(title = xtitle), yaxis = dict(title = ytitle),

             )

    fig = dict(data=data, layout=layout)

    iplot(fig, filename='lines')
count_all = historical_trans_df.groupby('dayofweek')['purchase_amount'].agg(['sum'])

count_all.columns = ["Total"]

count_all = count_all.sort_index()

plot_scatter_data(count_all['Total'],'Day of week', 'Total','Total sum of purchase per day of week','green')
count_all = historical_trans_df.groupby('weekofyear')['purchase_amount'].agg(['sum'])

count_all.columns = ["Total"]

count_all = count_all.sort_index()

plot_scatter_data(count_all['Total'],'Week of year', 'Total','Total sum of purchase per Week of Year','red')
count_all = historical_trans_df.groupby('month')['purchase_amount'].agg(['sum'])

count_all.columns = ["Total"]

count_all = count_all.sort_index()

plot_scatter_data(count_all['Total'],'Month', 'Total','Total sum of purchase per month','blue')
plot_bar(get_categories(new_merchant_trans_df,'category_1'), 

             'Category 1 distribution', 'Category 1', 'Number of records','gold')
plot_bar(get_categories(new_merchant_trans_df,'category_2'), 

             'Category 2 distribution', 'Category 2', 'Number of records','tomato')
plot_bar(get_categories(new_merchant_trans_df,'category_3'), 

             'Category 3 distribution', 'Category 3', 'Number of records','magenta')
plot_bar(get_categories(new_merchant_trans_df,'city_id'), 

             'City ID distribution', 'City ID', 'Number of records','brown')
plot_bar(get_categories(new_merchant_trans_df,'merchant_category_id'), 

             'Merchant category ID distribution', 'Merchant category ID', 'Number of records','green')
plot_bar(get_categories(new_merchant_trans_df,'state_id'), 

             'State ID distribution', 'State ID', 'Number of records','darkblue')
plot_bar(get_categories(new_merchant_trans_df,'subsector_id'), 

             'Subsector ID distribution', 'Subsector ID', 'Number of records','darkgreen')
new_merchant_trans_df['purchase_date'] = pd.to_datetime(new_merchant_trans_df['purchase_date'])

new_merchant_trans_df['month'] = new_merchant_trans_df['purchase_date'].dt.month

new_merchant_trans_df['dayofweek'] = new_merchant_trans_df['purchase_date'].dt.dayofweek

new_merchant_trans_df['weekofyear'] = new_merchant_trans_df['purchase_date'].dt.weekofyear
count_all = new_merchant_trans_df.groupby('month')['purchase_amount'].agg(['sum'])

count_all.columns = ["Total"]

count_all = count_all.sort_index()

plot_scatter_data(count_all['Total'],'Month', 'Total','Total sum of purchase per month','red')
count_all = new_merchant_trans_df.groupby('dayofweek')['purchase_amount'].agg(['sum'])

count_all.columns = ["Total"]

count_all = count_all.sort_index()

plot_scatter_data(count_all['Total'],'Day of week', 'Total','Total sum of purchase per day of week','magenta')
count_all = new_merchant_trans_df.groupby('weekofyear')['purchase_amount'].agg(['sum'])

count_all.columns = ["Total"]

count_all = count_all.sort_index()

plot_scatter_data(count_all['Total'],'Week of year', 'Total','Total sum of purchase per week of year','darkblue')
def plot_purchase_amount_distribution(data_df, var):

    hist_data = []

    varall = list(data_df.groupby([var])[var].nunique().index)

    for i, varcrt in enumerate(varall):

        classcrt = np.log(data_df[data_df[var] == varcrt]['purchase_amount'] + 1)

        hist_data.append(classcrt)

    fig = ff.create_distplot(hist_data, varall, show_hist=False, show_rug=False)

    fig['layout'].update(title='Purchase amount (log) variable density plot group by {}'.format(var), xaxis=dict(title='log(purchase_amount + 1)'))

    iplot(fig, filename='dist_only')
plot_purchase_amount_distribution(new_merchant_trans_df,'category_1')
plot_purchase_amount_distribution(new_merchant_trans_df,'category_2')
plot_purchase_amount_distribution(new_merchant_trans_df,'category_3')
plot_purchase_amount_distribution(new_merchant_trans_df,'state_id')
merchant_df.head(3)
plot_bar(get_categories(merchant_df,'merchant_category_id'), 

             'Merchant category ID distribution', 'Merchant category ID', 'Number of records','darkblue')
plot_bar(get_categories(merchant_df,'subsector_id'), 

             'Subsector ID distribution', 'Subsector ID', 'Number of records','blue')
plot_bar(get_categories(merchant_df,'category_1'), 

             'Category 1 distribution', 'Category 1', 'Number of records','lightblue')
plot_bar(get_categories(merchant_df,'category_2'), 

             'Category 2 distribution', 'Category 2', 'Number of records','lightgreen')
plot_bar(get_categories(merchant_df,'category_4'), 

             'Category 4 distribution', 'Category 4', 'Number of records','tomato')
plot_bar(get_categories(merchant_df,'most_recent_sales_range'), 

             'Most recent sales range distribution', 'Most recent sales range', 'Number of records','red')
plot_bar(get_categories(merchant_df,'most_recent_purchases_range'), 

             'Most recent sales purchases distribution', 'Most recent purchases range', 'Number of records','magenta')
plot_bar(get_categories(merchant_df,'city_id'), 

             'City ID distribution', 'City ID', 'Number of records','brown')
plot_bar(get_categories(merchant_df,'state_id'), 

             'State ID distribution', 'State ID', 'Number of records','orange')
def plot_distribution(df,feature,color):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12,6))

    s = sns.boxplot(ax = ax1, data = df[feature].dropna(),color=color,showfliers=True)

    s.set_title("Distribution of %s (with outliers)" % feature)

    s = sns.boxplot(ax = ax2, data = df[feature].dropna(),color=color,showfliers=False)

    s.set_title("Distribution of %s (no outliers)" % feature)

    plt.show()   
plot_distribution(merchant_df, "numerical_1", "blue")
plot_distribution(merchant_df, "numerical_2", "green")
plot_distribution(merchant_df, "avg_sales_lag3", "blue")
plot_distribution(merchant_df, "avg_sales_lag6", "green")
plot_distribution(merchant_df, "avg_sales_lag12", "green")
def get_logger():

    FORMAT = '[%(levelname)s]%(asctime)s:%(name)s:%(message)s'

    logging.basicConfig(format=FORMAT)

    logger = logging.getLogger('main')

    logger.setLevel(logging.DEBUG)

    return logger
# reduce memory

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

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))



    return df
logger = get_logger()

#process NAs

logger.info('Start processing NAs')

#process NA2 for transactions

for df in [historical_trans_df, new_merchant_trans_df]:

    df['category_2'].fillna(1.0,inplace=True)

    df['category_3'].fillna('A',inplace=True)

    df['merchant_id'].fillna('M_ID_00a6ca8a8a',inplace=True)

    df['installments'].replace(-1, np.nan,inplace=True)

    df['installments'].replace(999, np.nan,inplace=True)

#define function for aggregation

def create_new_columns(name,aggs):

    return [name + '_' + k + '_' + agg for k in aggs.keys() for agg in aggs[k]]
logger.info('process historical and new merchant datasets')

for df in [historical_trans_df, new_merchant_trans_df]:

    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    df['year'] = df['purchase_date'].dt.year

    df['weekofyear'] = df['purchase_date'].dt.weekofyear

    df['month'] = df['purchase_date'].dt.month

    df['dayofweek'] = df['purchase_date'].dt.dayofweek

    df['weekend'] = (df.purchase_date.dt.weekday >=5).astype(int)

    df['hour'] = df['purchase_date'].dt.hour

    df['authorized_flag'] = df['authorized_flag'].map({'Y':1, 'N':0})

    df['category_1'] = df['category_1'].map({'Y':1, 'N':0}) 

    df['category_3'] = df['category_3'].map({'A':0, 'B':1, 'C':2}) 

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']

logger.info('new features historical and new merchant datasets')

for df in [historical_trans_df, new_merchant_trans_df]:

    df['price'] = df['purchase_amount'] / df['installments']

    df['Christmas_Day_2017']=(pd.to_datetime('2017-12-25')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    df['Children_day_2017']=(pd.to_datetime('2017-10-12')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    df['Black_Friday_2017']=(pd.to_datetime('2017-11-24') - df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    df['Mothers_Day_2018']=(pd.to_datetime('2018-05-13')-df['purchase_date']).dt.days.apply(lambda x: x if x > 0 and x < 100 else 0)

    df['month_diff'] = ((datetime.datetime.today() - df['purchase_date']).dt.days)//30

    df['month_diff'] += df['month_lag']

    df['duration'] = df['purchase_amount']*df['month_diff']

    df['amount_month_ratio'] = df['purchase_amount']/df['month_diff']

logger.info('reduce memory usage for historical trans')

historical_trans_df = reduce_mem_usage(historical_trans_df)

logger.info('reduce memory usage for new merchant trans')

new_merchant_trans_df = reduce_mem_usage(new_merchant_trans_df)
#define aggregations with historical_trans_df

logger.info('Aggregate historical trans')

aggs = {}



for col in ['subsector_id','merchant_id','merchant_category_id', 'state_id', 'city_id']:

    aggs[col] = ['nunique']

for col in ['month', 'hour', 'weekofyear', 'dayofweek']:

    aggs[col] = ['nunique', 'mean', 'min', 'max']

    

aggs['purchase_amount'] = ['sum','max','min','mean','var', 'std']

aggs['installments'] = ['sum','max','min','mean','var', 'std']

aggs['purchase_date'] = ['max','min', 'nunique']

aggs['month_lag'] = ['max','min','mean','var','nunique']

aggs['month_diff'] = ['mean', 'min', 'max', 'var','nunique']

aggs['authorized_flag'] = ['sum', 'mean', 'nunique']

aggs['weekend'] = ['sum', 'mean', 'nunique']

aggs['year'] = ['nunique', 'mean']

aggs['category_1'] = ['sum', 'mean', 'min', 'max', 'nunique', 'std']

aggs['category_2'] = ['sum', 'mean', 'min', 'nunique', 'std']

aggs['category_3'] = ['sum', 'mean', 'min', 'nunique', 'std']

aggs['card_id'] = ['size', 'count']

aggs['Christmas_Day_2017'] = ['mean']

aggs['Children_day_2017'] = ['mean']

aggs['Black_Friday_2017'] = ['mean']

aggs['Mothers_Day_2018'] = ['mean']



for col in ['category_2','category_3']:

    historical_trans_df[col+'_mean'] = historical_trans_df.groupby([col])['purchase_amount'].transform('mean')

    historical_trans_df[col+'_min'] = historical_trans_df.groupby([col])['purchase_amount'].transform('min')

    historical_trans_df[col+'_max'] = historical_trans_df.groupby([col])['purchase_amount'].transform('max')

    historical_trans_df[col+'_sum'] = historical_trans_df.groupby([col])['purchase_amount'].transform('sum')

    historical_trans_df[col+'_std'] = historical_trans_df.groupby([col])['purchase_amount'].transform('std')

    aggs[col+'_mean'] = ['mean']    



new_columns = create_new_columns('hist',aggs)

historical_trans_group_df = historical_trans_df.groupby('card_id').agg(aggs)

historical_trans_group_df.columns = new_columns

historical_trans_group_df.reset_index(drop=False,inplace=True)

historical_trans_group_df['hist_purchase_date_diff'] = (historical_trans_group_df['hist_purchase_date_max'] - historical_trans_group_df['hist_purchase_date_min']).dt.days

historical_trans_group_df['hist_purchase_date_average'] = historical_trans_group_df['hist_purchase_date_diff']/historical_trans_group_df['hist_card_id_size']

historical_trans_group_df['hist_purchase_date_uptonow'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_max']).dt.days

historical_trans_group_df['hist_purchase_date_uptomin'] = (datetime.datetime.today() - historical_trans_group_df['hist_purchase_date_min']).dt.days



logger.info('reduce memory usage for historical trans')

historical_trans_df = reduce_mem_usage(historical_trans_df)



logger.info('Completed aggregate historical trans')
#merge with train, test

train_df = train_df.merge(historical_trans_group_df,on='card_id',how='left')

test_df = test_df.merge(historical_trans_group_df,on='card_id',how='left')

#cleanup memory

del historical_trans_group_df; gc.collect()
#define aggregations with new_merchant_trans_df 

logger.info('Aggregate new merchant trans')

aggs = {}

for col in ['subsector_id','merchant_id','merchant_category_id','state_id', 'city_id']:

    aggs[col] = ['nunique']

for col in ['month', 'hour', 'weekofyear', 'dayofweek']:

    aggs[col] = ['nunique', 'mean', 'min', 'max']



    

aggs['purchase_amount'] = ['sum','max','min','mean','var','std']

aggs['installments'] = ['sum','max','min','mean','var','std']

aggs['purchase_date'] = ['max','min', 'nunique']

aggs['month_lag'] = ['max','min','mean','var', 'nunique']

aggs['month_diff'] = ['mean', 'max', 'min', 'var','nunique']

aggs['weekend'] = ['sum', 'mean', 'nunique']

aggs['year'] = ['nunique', 'mean']

aggs['category_1'] = ['sum', 'mean', 'min', 'nunique']

aggs['category_2'] = ['sum', 'mean', 'min', 'nunique']

aggs['category_3'] = ['sum', 'mean', 'min', 'nunique']

aggs['card_id'] = ['size', 'count']

aggs['Christmas_Day_2017'] = ['mean']

aggs['Children_day_2017'] = ['mean']

aggs['Black_Friday_2017'] = ['mean']

aggs['Mothers_Day_2018'] = ['mean']



for col in ['category_2','category_3']:

    new_merchant_trans_df[col+'_mean'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('mean')

    new_merchant_trans_df[col+'_min'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('min')

    new_merchant_trans_df[col+'_max'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('max')

    new_merchant_trans_df[col+'_sum'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('sum')

    new_merchant_trans_df[col+'_std'] = new_merchant_trans_df.groupby([col])['purchase_amount'].transform('std')

    aggs[col+'_mean'] = ['mean']



new_columns = create_new_columns('new_hist',aggs)

new_merchant_trans_group_df = new_merchant_trans_df.groupby('card_id').agg(aggs)

new_merchant_trans_group_df.columns = new_columns

new_merchant_trans_group_df.reset_index(drop=False,inplace=True)

new_merchant_trans_group_df['new_hist_purchase_date_diff'] = (new_merchant_trans_group_df['new_hist_purchase_date_max'] - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days

new_merchant_trans_group_df['new_hist_purchase_date_average'] = new_merchant_trans_group_df['new_hist_purchase_date_diff']/new_merchant_trans_group_df['new_hist_card_id_size']

new_merchant_trans_group_df['new_hist_purchase_date_uptonow'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_max']).dt.days

new_merchant_trans_group_df['new_hist_purchase_date_uptomin'] = (datetime.datetime.today() - new_merchant_trans_group_df['new_hist_purchase_date_min']).dt.days



logger.info('reduce memory usage for new merchant trans')

new_merchant_trans_df = reduce_mem_usage(new_merchant_trans_df)



logger.info('Completed aggregate new merchant trans')
#merge with train, test

train_df = train_df.merge(new_merchant_trans_group_df,on='card_id',how='left')

test_df = test_df.merge(new_merchant_trans_group_df,on='card_id',how='left')

#clean-up memory

del new_merchant_trans_group_df; gc.collect()

del historical_trans_df; gc.collect()

del new_merchant_trans_df; gc.collect()
#process train

logger.info('Process train - outliers')

train_df['outliers'] = 0

train_df.loc[train_df['target'] < -30, 'outliers'] = 1

outls = train_df['outliers'].value_counts()

print("Outliers: {}".format(outls))

logger.info('Process train and test')

## process both train and test

for df in [train_df, test_df]:

    df['first_active_month'] = pd.to_datetime(df['first_active_month'])

    df['dayofweek'] = df['first_active_month'].dt.dayofweek

    df['weekofyear'] = df['first_active_month'].dt.weekofyear

    df['dayofyear'] = df['first_active_month'].dt.dayofyear

    df['quarter'] = df['first_active_month'].dt.quarter

    df['is_month_start'] = df['first_active_month'].dt.is_month_start

    df['month'] = df['first_active_month'].dt.month

    df['elapsed_time'] = (datetime.datetime.today() - df['first_active_month']).dt.days

    df['hist_first_buy'] = (df['hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['hist_last_buy'] = (df['hist_purchase_date_max'] - df['first_active_month']).dt.days

    df['new_hist_first_buy'] = (df['new_hist_purchase_date_min'] - df['first_active_month']).dt.days

    df['new_hist_last_buy'] = (df['new_hist_purchase_date_max'] - df['first_active_month']).dt.days

    for f in ['hist_purchase_date_max','hist_purchase_date_min','new_hist_purchase_date_max',\

                     'new_hist_purchase_date_min']:

        df[f] = df[f].astype(np.int64) * 1e-9

    df['card_id_total'] = df['new_hist_card_id_size']+df['hist_card_id_size']

    df['card_id_cnt_total'] = df['new_hist_card_id_count']+df['hist_card_id_count']

    df['purchase_amount_total'] = df['new_hist_purchase_amount_sum']+df['hist_purchase_amount_sum']

    df['purchase_amount_mean'] = df['new_hist_purchase_amount_mean']+df['hist_purchase_amount_mean']

    df['purchase_amount_max'] = df['new_hist_purchase_amount_max']+df['hist_purchase_amount_max']



    for f in ['feature_1','feature_2','feature_3']:

        order_label = train_df.groupby([f])['outliers'].mean()

        df[f] = df[f].map(order_label)



    df['feature_sum'] = df['feature_1'] + df['feature_2'] + df['feature_3']

    df['feature_mean'] = df['feature_sum']/3

    df['feature_max'] = df[['feature_1', 'feature_2', 'feature_3']].max(axis=1)

    df['feature_min'] = df[['feature_1', 'feature_2', 'feature_3']].min(axis=1)

    df['feature_var'] = df[['feature_1', 'feature_2', 'feature_3']].std(axis=1)



    

##

train_columns = [c for c in train_df.columns if c not in ['card_id', 'first_active_month','target','outliers']]

target = train_df['target']

del train_df['target']

logger.info('Completed process train')
#model

##model params

logger.info('Prepare model')

param = {'num_leaves': 51,

         'min_data_in_leaf': 35, 

         'objective':'regression',

         'max_depth': -1,

         'learning_rate': 0.008,

         "boosting": "gbdt",

         "feature_fraction": 0.85,

         "bagging_freq": 1,

         "bagging_fraction": 0.82,

         "bagging_seed": 42,

         "metric": 'rmse',

         "lambda_l1": 0.11,

         "verbosity": -1,

         "nthread": 4,

         "random_state": 2019}

#prepare fit model with cross-validation

folds = StratifiedKFold(n_splits=9, shuffle=True, random_state=2019)

oof = np.zeros(len(train_df))

predictions = np.zeros(len(test_df))

feature_importance_df = pd.DataFrame()

#run model

logger.info('Start running model')

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df,train_df['outliers'].values)):

    strLog = "Fold {}".format(fold_)

    print(strLog)

    trn_data = lgb.Dataset(train_df.iloc[trn_idx][train_columns], label=target.iloc[trn_idx])#, categorical_feature=categorical_feats)

    val_data = lgb.Dataset(train_df.iloc[val_idx][train_columns], label=target.iloc[val_idx])#, categorical_feature=categorical_feats)



    num_round = 10000

    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=200, early_stopping_rounds = 150)

    oof[val_idx] = clf.predict(train_df.iloc[val_idx][train_columns], num_iteration=clf.best_iteration)

    #feature importance

    fold_importance_df = pd.DataFrame()

    fold_importance_df["Feature"] = train_columns

    fold_importance_df["importance"] = clf.feature_importance()

    fold_importance_df["fold"] = fold_ + 1

    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #predictions

    predictions += clf.predict(test_df[train_columns], num_iteration=clf.best_iteration) / folds.n_splits

    logger.info(strLog)

    

strRMSE = "".format(np.sqrt(mean_squared_error(oof, target)))

print(strRMSE)
##plot the feature importance

logger.info("Feature importance plot")

cols = (feature_importance_df[["Feature", "importance"]]

        .groupby("Feature")

        .mean()

        .sort_values(by="importance", ascending=False)[:1000].index)

best_features = feature_importance_df.loc[feature_importance_df.Feature.isin(cols)]

plt.figure(figsize=(14,26))

sns.barplot(x="importance", y="Feature", data=best_features.sort_values(by="importance",ascending=False))

plt.title('LightGBM Features (averaged over folds)')

plt.tight_layout()

plt.savefig('lgbm_importances.png')
##submission

logger.info("Prepare submission")

sub_df = pd.DataFrame({"card_id":test_df["card_id"].values})

sub_df["target"] = predictions

sub_df.to_csv("submission.csv", index=False)