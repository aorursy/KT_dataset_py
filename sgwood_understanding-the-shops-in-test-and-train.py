import sys

import itertools

from pathlib import Path 

competition_data = Path('../input/competitive-data-science-predict-future-sales')

additional_data = Path('../input/russian-cities-population-wikipedia')

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from IPython.display import display, Markdown
# The shops and items in the test set for which we must make predictions

shops_in_test=pd.read_csv(competition_data/'test.csv',usecols=['shop_id']).shop_id.unique()



# The sales data

daily_sales=pd.read_csv(competition_data/'sales_train.csv',usecols=['shop_id','date_block_num','item_cnt_day'])


def plot_shop_trace(sales_data):

    

    all_sales_shops=sales_data.shop_id.unique()

    sales_by_shop_grid = sales_data.groupby(['shop_id','date_block_num']).sum().reset_index().pivot(index='shop_id',columns='date_block_num',values='item_cnt_day')

    sales_by_shop_l6m = sales_by_shop_grid.iloc[:,-6:].sum(axis=1)

    sales_by_shop_l6m2y = sales_by_shop_grid.iloc[:,-12:-6].sum(axis=1)

    sales_by_shop_y1 = sales_by_shop_grid.iloc[:,0:-12].sum(axis=1)



    yr1_shops = sales_by_shop_l6m[sales_by_shop_y1 > 0]

    yr2_shops = sales_by_shop_l6m[sales_by_shop_l6m2y > 0]

    live_shops = sales_by_shop_l6m[sales_by_shop_l6m > 0]



    f,ax=plt.subplots(1,1,figsize=(16,8))



    [ax.text(_x-0.22, 0.95, str(_x)) for _x in all_sales_shops];



    ax.text(60,0.8,"Shops with sales between prior to Oct 2014")

    [ax.axvline(_x, linewidth=4, ymin=0.7, ymax=0.9, color='#ac0f00') for _x in yr1_shops.index.unique()];



    ax.text(60,0.63,"Shops with sales between Oct 2014 and May 2015")

    [ax.axvline(_x, linewidth=4, ymin=0.6, ymax=0.67, color='#fc4f30') for _x in yr2_shops.index.unique()];



    ax.text(60,0.53,"Shops with sales in last 6 months")

    [ax.axvline(_x, linewidth=4, ymin=0.5, ymax=0.57, color='#e5ae38') for _x in live_shops.index.unique()];



    ax.text(60,0.3,"Shops we need to make predictions for")

    [ax.axvline(_x, linewidth=4, ymin=0.2, ymax=0.4, color='#30a2da') for _x in shops_in_test];



    [ax.text(_x-0.22, 0.1, str(_x)) for _x in shops_in_test];



    ax.set_axis_off(); 

    return ax



plot_shop_trace(daily_sales);
# shops:

# some shops look near duplicates from name...

# shop_id                    shop_name

# 10                         Жуковский ул. Чкалова 39м?

# 11                         Жуковский ул. Чкалова 39м²

# shop_id=10 is in the test set, so make occurances of 11 into 10s

daily_sales['shop_id']=daily_sales.shop_id.apply(lambda id: 10 if id==11 else id)



# combination of names and consecutive sales record suggests the following are same shop

# choose the later label to align with later months.

daily_sales['shop_id']=daily_sales.shop_id.apply(lambda id: 57 if id==0 else id)

daily_sales['shop_id']=daily_sales.shop_id.apply(lambda id: 58 if id==1 else id)



# redraw the trace plot

plot_shop_trace(daily_sales)
def extract_city_name(s):

    return s.lower().replace('!','').replace('. ', '.').replace(',','').replace('”','').replace('“','').replace('\"','').strip().split(' ')[0]



shop_meta=pd.read_csv(competition_data/'shops.csv')

shop_meta['extracted_city'] = shop_meta.shop_name.apply(lambda n: extract_city_name(n))



# a table taken from wikipedia - list of russian cities

# https://en.wikipedia.org/wiki/List_of_cities_and_towns_in_Russia_by_population

ru_cities=pd.read_csv(additional_data/'russian_city_populations.csv')

ru_cities['matching_city']=ru_cities.Russian.apply(lambda s: s.strip().lower())

shop_meta = shop_meta.merge(ru_cities[['matching_city','Population(2010 Census)[3]']], how='left', left_on='extracted_city', right_on='matching_city')  



# There are some that are not found. Some of them are translation errors that can be fixed manually



# 'интернет-магазин = 'online store'

# 'сергиев = is a city Sergiyev Posad (space problem)

# 'спб = spb

# 'н.новгород is a city Nizhny Novgorod (abbr problem) => Нижний Новгород

# 'выездная = 'exit'

# 'цифровой = 'digital'

# 'адыгея = The Republic of Adygea - assume captial city Майкоп

# 'ростовнадону' = is a city Rostov-on-Don]

# 'СПб' --> abbr for St Petersburg



# 3 of them do not seem to translate to city locations like the rest. Here are the literal translations:

special_shop_translations = {

    "Выездная Торговля":"Outbound Trade",

    "Интернет-магазин ЧС":"Emergency Shop Online",

    "Цифровой склад 1С-Онлайн":"Digital warehouse 1C-Online"

}



# fix the cities not found

mislabelled_cities={'спб':'санкт-петербург','сергиев':'Сергиев Посад','н.новгород':'Нижний Новгород','адыгея':'Майкоп','ростовнадону':'Ростов-на-Дону'}

shop_meta['matching_city']=shop_meta.apply(lambda r: mislabelled_cities[r['extracted_city']].lower() if r['extracted_city'] in mislabelled_cities.keys() else r['matching_city'], axis=1).astype('category')



# add a column to indicate whether we think this is a 'city' / 'normal' shop, or, one of the 'special' shops

shop_meta['shop_type']=shop_meta.apply(lambda r: 'city' if type(r['matching_city'])==str else special_shop_translations[r['shop_name']], axis=1)

r={'city':0,'Outbound Trade':1,'Emergency Shop Online':2,'Digital warehouse 1C-Online':3};

shop_meta['shop_type_numeric']=shop_meta.shop_type.apply(lambda s: r[s]).astype('category')



del shop_meta['extracted_city']; del shop_meta['Population(2010 Census)[3]']

def plot_shop_trace2(sales_data, fs=(16,8)):

    

    all_sales_shops=sales_data.shop_id.unique()

    sales_by_shop_grid = sales_data.groupby(['shop_id','date_block_num']).sum().reset_index().pivot(index='shop_id',columns='date_block_num',values='item_cnt_day')

    sales_by_shop_l1m = sales_by_shop_grid.iloc[:,-1:].sum(axis=1)

    sales_by_shop_l6m = sales_by_shop_grid.iloc[:,-6:].sum(axis=1)

    sales_by_shop_l6m2y = sales_by_shop_grid.iloc[:,-12:-6].sum(axis=1)

    

    sales_by_shop_y1 = sales_by_shop_grid.iloc[:,0:-12].sum(axis=1)



    yr1_shops = sales_by_shop_l6m[sales_by_shop_y1 > 0]

    yr2_shops = sales_by_shop_l6m[sales_by_shop_l6m2y > 0]

    live_shops = sales_by_shop_l6m[sales_by_shop_l6m > 0]

    vlive_shops = sales_by_shop_l6m[sales_by_shop_l1m > 0]



    ranked=sales_by_shop_l1m.sort_values(ascending=False).reset_index().reset_index()

    ranked.columns=['ranking','shop_id','sales']

    ranked=ranked.set_index('shop_id')

    

    FS=12

    

    f,ax=plt.subplots(1,1,figsize=fs)



    #ax.text(58,0.95,"shop_ids:  in training data (sales_train.csv)",fontsize=FS)

    [ax.axvline(ranked.loc[_x].ranking, linewidth=1, ymin=-0.1, ymax=0.95, color='#eeeeee') for _x in all_sales_shops];

    [ax.text(ranked.loc[_x].ranking-(0.25*len(str(_x))), 0.95, str(_x),fontsize=12) for _x in all_sales_shops];



    #ax.text(61,0.8,"with sales between prior to Oct 2014",fontsize=FS)

    [ax.axvline(ranked.loc[_x].ranking, linewidth=4, ymin=0.7, ymax=0.9, color='#ac0f00') for _x in yr1_shops.index.unique()];



    #ax.text(61,0.63,"with sales between Oct 2014 and May 2015",fontsize=FS)

    [ax.axvline(ranked.loc[_x].ranking, linewidth=4, ymin=0.6, ymax=0.68, color='#fc4f30') for _x in yr2_shops.index.unique()];



    #ax.text(61,0.55,"with sales in last 6 months",fontsize=FS)

    [ax.axvline(ranked.loc[_x].ranking, linewidth=4, ymin=0.53, ymax=0.58, color='#e5ae38') for _x in live_shops.index.unique()];



    #ax.text(61,0.48,"with sales in last month",fontsize=FS)

    [ax.axvline(ranked.loc[_x].ranking, linewidth=4, ymin=0.49, ymax=0.51, color='g') for _x in vlive_shops.index.unique()];



    #ax.text(58,0.0,"shop_ids we need to make predictions for",fontsize=FS)

    shop_meta['stc']=shop_meta.shop_type_numeric.apply(lambda v: '#30a2da' if v==0 else '#6d904f' )

    rel_sales=(sales_by_shop_l1m.sort_values()/sales_by_shop_l1m.max()).clip(0,100000)

    [ax.axvline(ranked.loc[_x].ranking, linewidth=4, ymin=0.0, ymax=0.4*rel_sales.loc[_x], color=shop_meta.set_index('shop_id').loc[_x,'stc']) for _x in shops_in_test];



    [ax.text(ranked.loc[_x].ranking-(0.25*len(str(_x))), -0.075, str(_x), fontsize=12) for _x in shops_in_test];

    ax.set_axis_off(); 

    

    return ax
ax=plot_shop_trace2(daily_sales, fs=(20,6))

ax.set_title("Mapping shop in test item file to shops in sales data.\n",fontsize=16);

plt.savefig('foo.png')