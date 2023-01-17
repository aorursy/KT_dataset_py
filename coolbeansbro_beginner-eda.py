import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from statsmodels.tsa.seasonal import seasonal_decompose
CA_train = pd.read_pickle('../input/m5-stuffs/CA_train.pkl')
grouped = CA_train.groupby(['cat_id','date'])['value'].sum()
group_shop = CA_train.groupby(['store_id','date'])['value'].sum()
indexing = np.array(grouped.index.get_level_values(level=0).unique().categories)
grouping_catnstore = CA_train.groupby(['store_id', 'cat_id','date'])['value'].sum()

idx = pd.IndexSlice

grouping_catnstore = grouping_catnstore.loc[idx['CA_1':'CA_4',:]].copy()
def plot_full(group_object,list_group, time_period, time):

    plt.figure(figsize=(15,6))

    for i in list_group:

        group_object[i].plot(x='date', y = 'value',label = i)

        if time:

            group_object[i].rolling(time_period).mean().plot(x='date',y='value', label = 'ROLLING MEAN ' + str(i))

    for i in list_group:

        plt.axhline(group_object[i].mean(),color = 'red')

        if time:

            plt.text(pd.Timestamp(2011,2,1),group_object[i].mean()+200, str(group_object[i].mean()), color = 'red')



    plt.legend()

    plt.show()

    return
plot_full(grouped, indexing, '30D', True)
def plot_timeperiod(group_object,list_grp,start_y,start_m,start_d,end_y,end_m,end_d):

    plt.figure(figsize=(15,5))

    for i in list_grp:

        group_object[i][(group_object[i].index>pd.Timestamp(start_y,start_m,start_d)) & 

                   (group_object[i].index<=pd.Timestamp(end_y,end_m,end_d))].plot(x='date',y='value', 

                                                                         label = i)

    plt.legend()

    plt.show()
plot_timeperiod(grouped,indexing,2014,1,1,2016,1,1)
plot_timeperiod(grouped,indexing,2014,12,1,2015,1,1)
grouping_withoutdate = CA_train.groupby(['store_id','cat_id'])['value'].sum()
idx = pd.IndexSlice

grouping_withoutdate = grouping_withoutdate.loc[idx['CA_1':'CA_4',:]].copy()
def plot_items_shop(grouped_object):

    grouped_object.plot(kind='bar', rot = 0, figsize = (15,6))

    plt.show()

    return 
plot_items_shop(grouping_withoutdate.unstack())
def plot_composition(**kwargs):

    result = seasonal_decompose(**kwargs).plot()

    plt.show()

    return
def plotting_acf_pacf(title,object_wanted,lags):

    fig,axes = plt.subplots(1,2, figsize= (12,3))

    plt.figure(figsize=(7,2))

    fig = plot_acf(object_wanted, lags = lags,ax=axes[0])

    fig = plot_pacf(object_wanted,lags=lags, ax = axes[1])

    fig.suptitle(title)

    plt.show()

    return
model_wanted = {'model': 'additive', 'x': grouped.HOUSEHOLD.resample('M').mean()}

title = 'ACF, PACF plots for Household Items'
plot_composition(**model_wanted)
plotting_acf_pacf(title,grouped.HOUSEHOLD,50)
model_year = {'x': grouped.HOUSEHOLD[(grouped.HOUSEHOLD.index>pd.Timestamp(2015,1,1)) & 

               (grouped.HOUSEHOLD.index<=pd.Timestamp(2016,1,1))], 'model':'additive' }
plot_composition(**model_year)
title = 'ACF,PACF Plots for Food Items'

model_food = {'model': 'additive', 'x': grouped.FOODS.resample('M').mean()}

plot_composition(**model_food)
model_5_food = {'model': 'additive','x': grouped.FOODS[(grouped.FOODS.index>pd.Timestamp(2015,1,1)) & 

               (grouped.FOODS.index<=pd.Timestamp(2016,1,1))]}
plot_composition(**model_5_food)
plotting_acf_pacf(title,grouped.FOODS, 50)
title = 'ACF, PACF plots for Hobby items'

plotting_acf_pacf(title,grouped.HOBBIES, 50)
model_hobbies = {'x': grouped.HOBBIES.resample('M').mean(), 'model': 'additive'}
plot_composition(**model_hobbies)
index_shops = np.array(group_shop.index.get_level_values(level=0).categories)
index_shops = np.delete(index_shops,[4,5,6,7,8,9],None)
plot_full(group_shop, index_shops, '30D', True)
plot_timeperiod(group_shop, index_shops, 2015,12,1,2016,1,1)
model_shop_listing = []

for i,shop_name in enumerate(index_shops):

    model_shop_listing.append({'x': group_shop[shop_name].resample('M').mean(), 'model':'additive'})

    title = 'Plots for ' + shop_name

    plotting_acf_pacf(title,group_shop[shop_name],50)

    plot_composition(**model_shop_listing[i])

    
index_shops = grouping_catnstore.index.get_level_values(level=0).unique()

index_cat = grouping_catnstore.index.get_level_values(level=1).unique()

plt.figure(figsize = (12,6))

for i in index_shops:

    for j in index_cat:

        grouping_catnstore[i][j].plot(x='date',y='value', label = i+' ' + j)

        grouping_catnstore[i][j].rolling('30D').mean().plot(x = 'date', y= 'value', 

                                                            label = 'ROLLING MEAN' + i + ' ' + j )

    plt.legend()

    plt.show()
model_catnstore_listing = []

for i,shop_name in enumerate(index_shops):

    model_catnstore_listing.append([])

    for j,cat in enumerate(index_cat):

        model_catnstore_listing[i].append({'x': grouping_catnstore[shop_name][cat].resample('M').mean(),

                                         'model':'additive'})

        title = 'Plots for ' + shop_name + ' ' + cat

        plotting_acf_pacf(title,grouping_catnstore[shop_name][cat],50)

        plot_composition(**model_catnstore_listing[i][j])

        
sns.jointplot(CA_train['sell_price'], CA_train['value'])

plt.show()
grouped_agg = CA_train.groupby(['cat_id','date']).agg({

    'value':np.sum, 'sell_price':'mean'}).rename(columns={

    'value': 'sum_val', 'sell_price': 'avg_price'})
for i in grouped_agg.index.get_level_values(level=0).unique():

    plots = sns.jointplot(grouped_agg.loc[i].avg_price,grouped_agg.loc[i].sum_val)

    plots.fig.suptitle(i, y=0.85)

    plt.show()
sales_price = grouped_agg.groupby(['cat_id'] +[pd.Grouper(freq='W', level=-1)]).mean()
sales_price.rename(columns = {'sum_val':'avg_val'}, inplace = True)
from sklearn.preprocessing import MinMaxScaler

ss = MinMaxScaler()

transformed = ss.fit_transform(sales_price)
sales_price['avg_val'] = transformed[:,0]

sales_price['avg_price'] = transformed[:,1]
plt.figure(figsize=(15,8))

for i in sales_price.index.get_level_values(level=0).unique():

    sales_price.loc[i]['avg_val'].plot(x='date', y='avg_val', label = 'avg val' + ' ' +i)

    sales_price.loc[i]['avg_price'].plot(x='date', y='avg_price', label = 'avg price'+" " + i)

plt.legend()

plt.show()
snap_ca_difference = CA_train[CA_train['snap_CA']==1].copy()

snap_ca_difference = snap_ca_difference.groupby(['store_id','week'])['value'].sum()
idx = pd.IndexSlice

snap_ca_difference = snap_ca_difference.loc[idx['CA_1':'CA_4',:]].copy()

indexing_stores = snap_ca_difference.index.get_level_values(level=0).unique()
plot_full(snap_ca_difference,indexing_stores, 4, False)