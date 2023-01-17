import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tic

import gc
import lightgbm
from functools import partial
from hyperopt import fmin, hp, tpe, Trials, space_eval, STATUS_OK, STATUS_RUNNING
from plotly import __version__
%matplotlib inline

import plotly.tools as tls
from plotly.subplots import make_subplots
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.offline import iplot

import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot 

# Using plotly + cufflinks in offline mode
init_notebook_mode(connected=True)
cf.go_offline(connected=True)
cf.go_offline()

import seaborn as sns
import datetime
import random
import IPython
from IPython.core.interactiveshell import InteractiveShell

from itertools import product

%matplotlib inline
#later problem with cpu and ram
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
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    
    return df
#submission = pd.read_csv('../input/sample_submission.csv')
item_cat = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
sales = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
shops.head(5)
shops.shape
shops.describe()
shops.shop_id.unique()
item_cat.head(5)
item_cat.shape
item_cat.describe()
item_cat.item_category_id.unique()
items.head(5)
items.shape
items.describe()
items.item_id.unique().shape[0]
test.head(5)
test.shape
#submission.head(5)
sales.head(5)
sales.shape
sales.describe()
#correct format of time
sales.date    = pd.to_datetime(sales.date, format = '%d.%m.%Y')
#aggregate all data
sales_train = sales\
.join(items.set_index('item_id'), on='item_id')\
.join(item_cat.set_index('item_category_id'), on='item_category_id')\
.join(shops.set_index('shop_id'), on='shop_id')

sales_train.head(2).T
#test\
#.join(submission.set_index('ID'), on='ID').head()
dict_item_categories = ['Cinema - DVD', 'PC Games - Standard Editions',
                    'Music - Local Production CD', 'Games - PS3', 'Cinema - Blu-Ray',
                    'Games - XBOX 360', 'PC Games - Additional Editions', 'Games - PS4',
                    'Gifts - Stuffed Toys', 'Gifts - Board Games (Compact)',
                    'Gifts - Figures', 'Cinema - Blu-Ray 3D',
                    'Programs - Home and Office', 'Gifts - Development',
                    'Gifts - Board Games', 'Gifts - Souvenirs (on the hinge)',
                    'Cinema - Collection', 'Music - MP3', 'Games - PSP',
                    'Gifts - Bags, Albums, Mouse Pads', 'Gifts - Souvenirs',
                    'Books - Audiobooks', 'Gifts - Gadgets, robots, sports',
                    'Accessories - PS4', 'Games - PSVita',
                    'Books - Methodical materials 1C', 'Payment cards - PSN',
                    'PC Games - Digit', 'Games - Game Accessories', 'Accessories - XBOX 360',
                    'Accessories - PS3', 'Games - XBOX ONE', 'Music - Vinyl',
                    'Programs - 1C: Enterprise 8', 'PC Games - Collectible Editions',
                    'Gifts - Attributes', 'Service Tools',
                    'Music - branded production CD', 'Payment cards - Live!',
                    'Game consoles - PS4', 'Accessories - PSVita', 'Batteries',
                    'Music - Music Video', 'Game Consoles - PS3',
                    'Books - Comics, Manga', 'Game Consoles - XBOX 360',
                    'Books - Audiobooks 1C', 'Books - Digit',
                    'Payment cards (Cinema, Music, Games)', 'Gifts - Cards, stickers',
                    'Accessories - XBOX ONE', 'Pure media (piece)',
                    'Programs - Home and Office (Digital)', 'Programs - Educational',
                    'Game consoles - PSVita', 'Books - Artbooks, encyclopedias',
                    'Programs - Educational (Digit)', 'Accessories - PSP',
                    'Gaming consoles - XBOX ONE', 'Delivery of goods',
                    'Payment Cards - Live! (Figure) ',' Tickets (Figure) ',
                    'Music - Gift Edition', 'Service Tools - Tickets',
                    'Net media (spire)', 'Cinema - Blu-Ray 4K', 'Game consoles - PSP',
                    'Game Consoles - Others', 'Books - Audiobooks (Figure)',
                    'Gifts - Certificates, Services', 'Android Games - Digit',
                    'Programs - MAC (Digit)', 'Payment Cards - Windows (Digit)',
                    'Books - Business Literature', 'Games - PS2', 'MAC Games - Digit',
                    'Books - Computer Literature', 'Books - Travel Guides',
                    'PC - Headsets / Headphones', 'Books - Fiction',
                    'Books - Cards', 'Accessories - PS2', 'Game consoles - PS2',
                    'Books - Cognitive literature']

dict_shops = ['Moscow Shopping Center "Semenovskiy"', 
              'Moscow TRK "Atrium"', 
              "Khimki Shopping Center",
              'Moscow TC "MEGA Teply Stan" II', 
              'Yakutsk Ordzhonikidze, 56',
              'St. Petersburg TC "Nevsky Center"', 
              'Moscow TC "MEGA Belaya Dacha II"',
              'Voronezh (Plekhanovskaya, 13)', 
              'Yakutsk Shopping Center "Central"',
              'Chekhov SEC "Carnival"', 
              'Sergiev Posad TC "7Ya"',
              'Tyumen TC "Goodwin"',
              'Kursk TC "Pushkinsky"', 
              'Kaluga SEC "XXI Century"',
              'N.Novgorod Science and entertainment complex "Fantastic"',
              'Moscow MTRC "Afi Mall"',
              'Voronezh SEC "Maksimir"', 'Surgut SEC "City Mall"',
              'Moscow Shopping Center "Areal" (Belyaevo)', 'Krasnoyarsk Shopping Center "June"',
              'Moscow TK "Budenovsky" (pav.K7)', 'Ufa "Family" 2',
              'Kolomna Shopping Center "Rio"', 'Moscow Shopping Center "Perlovsky"',
              'Moscow Shopping Center "New Century" (Novokosino)', 'Omsk Shopping Center "Mega"',
              'Moscow Shop C21', 'Tyumen Shopping Center "Green Coast"',
              'Ufa TC "Central"', 'Yaroslavl shopping center "Altair"',
              'RostovNaDonu "Mega" Shopping Center', '"Novosibirsk Mega "Shopping Center',
              'Samara Shopping Center "Melody"', 'St. Petersburg TC "Sennaya"',
              "Volzhsky Shopping Center 'Volga Mall' ",
              'Vologda Mall "Marmelad"', 'Kazan TC "ParkHouse" II',
              'Samara Shopping Center ParkHouse', '1C-Online Digital Warehouse',
              'Online store of emergencies', 'Adygea Shopping Center "Mega"',
              'Balashikha shopping center "October-Kinomir"' , 'Krasnoyarsk Shopping center "Vzletka Plaza" ',
              'Tomsk SEC "Emerald City"', 'Zhukovsky st. Chkalov 39m? ',
              'Kazan Shopping Center "Behetle"', 'Tyumen SEC "Crystal"',
              'RostovNaDonu TRK "Megacenter Horizon"',
              '! Yakutsk Ordzhonikidze, 56 fran', 'Moscow TC "Silver House"',
              'Moscow TK "Budenovsky" (pav.A2)', "N.Novgorod SEC 'RIO' ",
              '! Yakutsk TTS "Central" fran', 'Mytishchi TRK "XL-3"',
              'RostovNaDonu TRK "Megatsentr Horizon" Ostrovnoy', 'Exit Trade',
              'Voronezh SEC City-Park "Grad"', "Moscow 'Sale'",
              'Zhukovsky st. Chkalov 39m² ',' Novosibirsk Shopping Mall "Gallery Novosibirsk"']
sales_train.item_category_name = sales_train.item_category_name.map(dict(zip(sales_train.item_category_name.value_counts().index, dict_item_categories)))
sales_train.shop_name = sales_train.shop_name.map(dict(zip(sales_train.shop_name.value_counts().index, dict_shops)))
sales_train.head().T
#change to comparable format and to month instead of day
pd.pivot_table(sales_train.loc[sales_train.date_block_num==33], index='item_id', columns='shop_id', values='item_cnt_day', aggfunc='sum')\
.clip(0,20).fillna(0)\
.unstack().to_frame('item_cnt_month').reset_index().head()

#preparation data
piv = np.log(pd.pivot_table(sales_train,
                            index='date',
                            columns=['shop_id','shop_name'],
                            values='item_cnt_day',
                            aggfunc='sum').fillna(0).clip(0,)+1).T

#plotting
max_width= 12
fig, ax = plt.subplots(figsize=(max_width,10))

_y = piv.index.get_level_values(0)[::-1]
_x = mdates.date2num(piv.columns)
    
ax.imshow(piv, aspect='auto', extent = [ _x[0],  _x[-1]+1,  _y[0]+1, _y[-1]], interpolation='none')

ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y - %m'))
ax.xaxis.set_minor_locator(mdates.MonthLocator())
ax.xaxis.set_minor_formatter(mdates.DateFormatter('%m'))

ax.yaxis.set_ticks(_y)
ax.set_yticklabels(piv.index.get_level_values(0)[::-1].astype('str') + ' ' + piv.index.get_level_values(1)[::-1],
                  fontsize=8, va='top')

fig.autofmt_xdate(which='both', rotation=90, ha='left')


plt.tight_layout()
plt.show()
sales_train.drop_duplicates(subset=['item_id', 'shop_id']).plot.scatter('item_id', 'shop_id', color='DarkBlue', s = 0.1)
test.drop_duplicates(subset=['item_id', 'shop_id']).plot.scatter('item_id', 'shop_id', color='DarkGreen', s = 0.1)
test.merge(sales_train, how='left', on=['shop_id', 'item_id']).isnull().sum()
test.shape[0]

def quantiles(df, columns):
    for name in columns:
        print(name + " quantiles")
        print(df[name].quantile([.01,.25,.5,.75,.99]))
        print("")
sales_train['total_amount'] = sales_train['item_price'] * sales_train['item_cnt_day']
quantiles(sales_train, ['item_cnt_day','item_price', 'total_amount'])
!pip install squarify
import squarify

color = ["#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(30)]
shop_name = sales_train["shop_name"].value_counts() #counting the values of shop names

print("Description most frequent countrys: ")
print(shop_name[:10]) #printing the 15 top most 

shop = round((sales_train["shop_name"].value_counts()[:20] \
                       / len(sales_train["shop_name"]) * 100),2)

plt.figure(figsize=(20,10))
g = squarify.plot(sizes=shop.values, label=shop.index, 
                  value=shop.values,
                  alpha=.8, color=color)
g.set_title("'TOP 20 Stores/Shop - % size of total",fontsize=20)
g.set_axis_off()
plt.show()
print("Percentual of total sold by each Shop")
print((sales_train.groupby('shop_name')['item_price'].sum().nlargest(25) / sales_train.groupby('shop_name')['item_price'].sum().sum() * 100)[:5])

sales_train.groupby('shop_name')['item_price'].sum().nlargest(25).plot(kind='bar',
                                                                     title='TOP 25 Shop Name by Total Amount Sold')
print("Percentual of total sold by each Shop")
print((sales_train.groupby('shop_name')['item_cnt_day'].sum().nlargest(25) / sales_train.groupby('shop_name')['item_cnt_day'].sum().sum() * 100)[:5])

sales_train.groupby('shop_name')['item_cnt_day'].sum().nlargest(25).plot(kind='bar',
                                                                       title='TOP 25 Shop Name by Total Amount Sold')
top_cats = sales_train.item_category_name.value_counts()[:15]

plt.figure(figsize=(15,20))

plt.subplot(311)
g1 = sns.countplot(x='item_category_name', 
                   data=sales_train[sales_train.item_category_name.isin(top_cats.index)])
g1.set_xticklabels(g1.get_xticklabels(),rotation=70)
g1.set_title("TOP 15 Principal Products Sold", fontsize=22)
g1.set_xlabel("")
g1.set_ylabel("Count", fontsize=18)

plt.subplot(312)
g2 = sns.boxplot(x='item_category_name', y='item_cnt_day', 
                   data=sales_train[sales_train.item_category_name.isin(top_cats.index)])
g2.set_xticklabels(g2.get_xticklabels(),rotation=70)
g2.set_title("Principal item_categories by Item Solds Log", fontsize=22)
g2.set_xlabel("")
g2.set_ylabel("Items Sold Log Distribution", fontsize=18)

plt.subplot(313)
g3 = sns.boxplot(x='item_category_name', y='total_amount', 
                   data=sales_train[sales_train.item_category_name.isin(top_cats.index)])
g3.set_xticklabels(g3.get_xticklabels(),rotation=70)
g3.set_title("Category Name by Total Amount Log", fontsize=22)
g3.set_xlabel("")
g3.set_ylabel("Total Amount Log", fontsize=18)

plt.subplots_adjust(wspace = 0.2, hspace = 0.8,top = 0.9)
plt.show()
sub_categorys_5000 = sales_train.sort_values('total_amount',
                                          ascending=False)[['item_category_name', 'item_name', 
                                                            'shop_name',
                                                            'item_cnt_day','item_price',
                                                            'total_amount']].head(5000)
sub_categorys_5000.head(10)
sub_categorys_5000 = sales_train.sort_values('item_price',
                                          ascending=False)[['item_category_name', 'item_name', 
                                                            'shop_name',
                                                            'item_cnt_day','item_price',
                                                            'total_amount']].head(5000)
sub_categorys_5000.head(10)
plt.figure(figsize=(14,26))

plt.subplot(311)
g = sns.countplot(x='shop_name', data=sub_categorys_5000)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Count of Shop Names in Top Bills ", fontsize=22)
g.set_xlabel('Shop Names', fontsize=18)
g.set_ylabel("Total Count in expensive bills", fontsize=18)

plt.subplot(312)
g = sns.countplot(x='item_category_name', data=sub_categorys_5000)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Count of Category Name in Top Bills", fontsize=22)
g.set_xlabel('Category Names', fontsize=18)
g.set_ylabel("Total Count in expensive bills", fontsize=18)

plt.subplot(313)
g = sns.boxenplot(x='item_category_name', y='item_cnt_day', data=sub_categorys_5000)
g.set_xticklabels(g.get_xticklabels(),rotation=90)
g.set_title("Count of Category Name in Top Bills by Total items Sold", fontsize=22)
g.set_xlabel('Most top Category Names', fontsize=18)
g.set_ylabel("Total sold distribution", fontsize=18)

plt.subplots_adjust(wspace = 0.2, hspace = 1.6,top = 0.9)

plt.show()
sales_train.item_price.plot()
sales_train[sales_train['item_price']>100000]
print("TOTAL REPRESENTATION OF TOP 5k Most Expensive orders: ", 
      f'{round((sub_categorys_5000.item_price.sum() / sales_train.item_price.sum()) * 100, 2)}%')
def cross_heatmap(df, cols, normalize=False, values=None, aggfunc=None):
    temp = cols
    cm = sns.light_palette("green", as_cmap=True)
    return pd.crosstab(df[temp[0]], df[temp[1]], 
                       normalize=normalize, values=values, aggfunc=aggfunc).style.background_gradient(cmap = cm)

cross_heatmap(sub_categorys_5000, ['item_category_name', 'item_cnt_day'])
sub_categorys_5000 = sales_train.sort_values('item_cnt_day',
                                          ascending=False)[['item_category_name', 'item_name', 
                                                            'shop_name',
                                                            'item_cnt_day','item_price',
                                                            'total_amount']].head(5000)
sub_categorys_5000.head(10)
sales_train.item_cnt_day.plot()
cross_heatmap(sales_train.sample(500000), ['item_category_name', 'shop_name'], 
              normalize='columns', aggfunc='sum', values=sales_train['total_amount'])
#fix ids 
sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
#some handpicked outliers: only include item price less than 100000 and amount of sales less or equal to 900
sales_train = sales_train[sales_train.item_price<100000]
sales_train = sales_train[sales_train.item_cnt_day<=900]

sales_train.head().T
sales_train['year']            = sales_train.date.dt.year
sales_train['month']           = sales_train.date.dt.month
sales_train['day']             = sales_train.date.dt.day
sales_train['day_of_year']     = sales_train.date.dt.dayofyear
sales_train['week_day']        = sales_train.date.dt.weekday
sales_train['activity_cnt']    = sales_train.item_cnt_day.abs()
sales_train['activity_rev']    = sales_train.activity_cnt*sales_train.item_price
sales_train['revenue']         = sales_train.item_cnt_day*sales_train.item_price
items = items\
.join(sales_train.groupby('item_id').agg(
    sale_start=('date','min'),
    sale_end=('date','max'),
    price_mean=('item_price','mean'),
    #sales_train_cnt=('item_cnt_day','sum'),
    item_total_revenue=('revenue','sum')))

#seting some static color options
color_op = ['#5527A0', '#BB93D7', '#834CF7', '#6C941E', '#93EAEA', '#7425FF', '#F2098A', '#7E87AC', 
            '#EBE36F', '#7FD394', '#49C35D', '#3058EE', '#44FDCF', '#A38F85', '#C4CEE0', '#B63A05', 
            '#4856BF', '#F0DB1B', '#9FDBD9', '#B123AC']
dates_temp = sales_train['date'].value_counts().to_frame().reset_index().sort_values('index') 
# renaming the columns to apropriate names
dates_temp = dates_temp.rename(columns = {"date" : "Total_Bills"}).rename(columns = {"index" : "date"})

# creating the first trace with the necessary parameters
trace = go.Scatter(x=dates_temp.date.astype(str), y=dates_temp.Total_Bills,
                    opacity = 0.8, line = dict(color = color_op[7]), name= 'Total tickets')

# Below we will get the total amount sold
dates_temp_sum = sales_train.groupby('date')['item_price'].sum().to_frame().reset_index()

# using the new dates_temp_sum we will create the second trace
trace1 = go.Scatter(x=dates_temp_sum.date.astype(str), line = dict(color = color_op[1]), name="Total Amount",
                        y=dates_temp_sum['item_price'], opacity = 0.8)

# Getting the total values by Transactions by each date
dates_temp_count = sales_train[sales_train['item_cnt_day'] > 0].groupby('date')['item_cnt_day'].sum().to_frame().reset_index()

# using the new dates_temp_count we will create the third trace
trace2 = go.Scatter(x=dates_temp_count.date.astype(str), line = dict(color = color_op[5]), name="Total Items Sold",
                        y=dates_temp_count['item_cnt_day'], opacity = 0.8)

#creating the layout the will allow us to give an title and 
# give us some interesting options to handle with the outputs of graphs
layout = dict(
    title= "Informations by Date",
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(step='all')
            ])
        ),
        rangeslider=dict(visible = True),
        type='date'
    )
)

# creating figure with the both traces and layout
fig = dict(data= [trace, trace1, trace2], layout=layout)

#rendering the graphs
iplot(fig) #it's an equivalent to plt.show()
sales_train.head().T
def generate_random_color():
    r = lambda: random.randint(0,255)
    return '#%02X%02X%02X' % (r(),r(),r())

#shared Xaxis parameter can make this graph look even better
fig = tls.make_subplots(rows=2, cols=1)

layout1 = cf.Layout(
    height=500,
    width=200
)
animal_color = generate_random_color()
fig1 = sales_train.groupby(['month'])['item_cnt_day'].count().iplot(kind='bar',barmode='stack',
                                                                  asFigure=True,showlegend=False,
                                                                  title='Total Items Sold By Month',
                                                                  xTitle='Months', yTitle='Total Items Sold',
                                                                  color = 'blue')
fig1['data'][0]['showlegend'] = False
fig.append_trace(fig1['data'][0], 1, 1)


fig2 = sales_train.groupby(['month'])['item_cnt_day'].sum().iplot(kind='bar',barmode='stack',
                                                                title='Total orders by Month',
                                                                xTitle='Months', yTitle='Total Orders',
                                                                asFigure=True, showlegend=False, 
                                                                color = 'blue')

#if we do not use the below line there will be two legend
fig2['data'][0]['showlegend'] = False


fig.append_trace(fig2['data'][0], 2, 1)

layout = dict(
    title= "Informations by Date",
    )

fig['layout']['height'] = 800
fig['layout']['width'] = 1000
fig['layout']['title'] = "TOTAL ORDERS AND TOTAL ITEMS BY MONTHS"
fig['layout']['yaxis']['title'] = "Total Items Sold"
fig['layout']['xaxis']['title'] = "Months"
fig['layout']

iplot(fig)
## Deleting the datasets that was used to explore the data
del sales_train
del test
del shops
del items
del item_cat

gc.collect()

## Importing the df's again to modelling

item_cat = pd.read_csv("../input/competitive-data-science-predict-future-sales/item_categories.csv")
items = pd.read_csv("../input/competitive-data-science-predict-future-sales/items.csv")
shops = pd.read_csv("../input/competitive-data-science-predict-future-sales/shops.csv")
sales_train = pd.read_csv("../input/competitive-data-science-predict-future-sales/sales_train.csv")
test = pd.read_csv("../input/competitive-data-science-predict-future-sales/test.csv")
#some handpicked outliers: only include item price less than 100000 and amount of sales less or equal to 900
sales_train = sales_train[sales_train.item_price<100000]
sales_train = sales_train[sales_train.item_cnt_day<=900]
#fix ids 
sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
median = sales_train[(sales_train.shop_id==32)&(sales_train.item_id==2973)&(sales_train.date_block_num==4)&(sales_train.item_price>0)].item_price.median()
sales_train.loc[sales_train.item_price<0, 'item_price'] = median
shops.head(2)
from sklearn.preprocessing import LabelEncoder

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]

item_cat['split'] = item_cat['item_category_name'].str.split('-')
item_cat['type'] = item_cat['split'].map(lambda x: x[0].strip())
item_cat['type_code'] = LabelEncoder().fit_transform(item_cat['type'])
# if subtype is nan then type
item_cat['subtype'] = item_cat['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
item_cat['subtype_code'] = LabelEncoder().fit_transform(item_cat['subtype'])
item_cat = item_cat[['item_category_id','type_code', 'subtype_code']]

items.drop(['item_name'], axis=1, inplace=True)
import time

# Creating the Matrix
matrix = []

# Column names
cols = ['date_block_num','shop_id','item_id']

# Creating the pairwise for each date_num_block
for i in range(34):
    # Filtering sales by each month
    sales = sales_train[sales_train.date_block_num==i]
    # Creating the matrix date_block, shop_id, item_id
    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
    
# Seting the matrix to dataframe
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

# Seting the features to int8 to reduce memory usage
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)
matrix['shop_id'] = matrix['shop_id'].astype(np.int8)
matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)
# Creating the revenue column
sales_train['revenue'] = sales_train['item_price'] *  sales_train['item_cnt_day']
# getting the total itens sold by each date_block for each shop_id and item_id pairs
group = sales_train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
# Renaming columns
group.columns = ['item_cnt_month']
# Reset the index 
group.reset_index(inplace=True)

# Merging the grouped column to our matrix
matrix = pd.merge(matrix, group, on=cols, how='left')
# Filling Na's and clipping the values to have range 0,20
# seting to float16 to reduce memory usage
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))

matrix.head()
gc.collect()
# Creating the date_block in df_test
test['date_block_num'] = 34

# Seting the df test columns to int8 to reduce memory usage
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
## Concatenating the test set into matrix and filling Na's with zero
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month
# merging the shops, items, and categories in matrix
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, item_cat, on=['item_category_id'], how='left')

# Seting the new columns to int8 to reduce memory
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

matrix.head()
# Function to calculate lag features
def lag_feature(df, lags, col):
    # Columns to get lag
    tmp = df[['date_block_num','shop_id','item_id',col]]
    # loop for each lag value in the list
    for i in lags:
        # Coping the df
        shifted = tmp.copy()
        # Creating the lag column
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        # getting the correct date_num_block to calculation
        shifted['date_block_num'] += i
        # merging the new column into the matrix
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
        
    return df
# Creating the lag columns 
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')

matrix.head()
# Getting the mean item_cnt_month by date_bock
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
# Renaming
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the grouped object into the matrix
matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
# creating the lag column to average itens solds
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
# Droping the date_avg_item_cnt
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
## Getting the mean item solds by date_blocks and item_id 
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
# Renaming column
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)

# Geting the lag feature to the new column
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
# Grouping the mean items sold by shop id for each date_block
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
# Renaming Columns
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the grouped into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)

# Geting the lag of the new column
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
## Getting the mean items sold by item_category_id for each date_block_num
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})

# Renaming column
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
## Getting the mean items sold by shop_id and item_category_id for each date_block_num
group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
## Getting the mean items sold by shop_id and subtype_code for each date_block_num
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
matrix = reduce_mem_usage(matrix)
## Getting the mean items sold by shop_id and item_category_id for each date_block_num
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
## Getting the mean items sold by city_code for each date_block_num
group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_city_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
## Getting the mean items sold by item_id and city_code for each date_block_num
group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
matrix = reduce_mem_usage(matrix)
## Getting the mean items sold by type_code for each date_block_num
group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
## Getting the mean items sold by subtype_code for each date_block_num
group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')

matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)

# Getting the lag of the new column
matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)

matrix.head()
del items
del shops
del item_cat
matrix = reduce_mem_usage(matrix)
# Getting the mean item price by item_id
group = sales_train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['item_id'], how='left')

matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

## Getting the mean item price by item_id for each date_block_num
group = sales_train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

# Merging the new grouped object into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

# Geting the lags of date item avg item price
lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

# seting the delta price lag for each lag price
for i in lags:
    matrix['delta_price_lag_'+str(i)] = \
        (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

# Selecting trends
def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0
    
# Applying the trend selection
matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

# Getting the features to drop
fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)
gc.collect()

matrix.head()
# Getting the revenue sum by shop_id and date_block
group = sales_train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

# Merging the new group into matrix
matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

# Getting the mean item price by item_id
group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
matrix['month'] = matrix['date_block_num'] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)
cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = str(row.item_id)+' '+str(row.shop_id)
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = row.date_block_num - last_date_block_num
        cache[key] = row.date_block_num         

cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num>last_date_block_num:
            matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
matrix = matrix[matrix.date_block_num > 11]
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)
matrix.info()
matrix.to_pickle('data.pkl')

del matrix
del cache
del group
del sales_train

# leave test for submission
gc.collect();
data = pd.read_pickle('data.pkl')
data = data[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'city_code', 'item_category_id',
             'type_code', 'subtype_code', 'item_cnt_month_lag_1', 'item_cnt_month_lag_2',
             'item_cnt_month_lag_3', 'item_cnt_month_lag_6', 'item_cnt_month_lag_12',
             'date_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_1', 'date_item_avg_item_cnt_lag_2',
             'date_item_avg_item_cnt_lag_3', 'date_item_avg_item_cnt_lag_6', 'date_item_avg_item_cnt_lag_12',
             'date_shop_avg_item_cnt_lag_1', 'date_shop_avg_item_cnt_lag_2', 'date_shop_avg_item_cnt_lag_3',
             'date_shop_avg_item_cnt_lag_6', 'date_shop_avg_item_cnt_lag_12', 'date_cat_avg_item_cnt_lag_1',
             'date_shop_cat_avg_item_cnt_lag_1', 'item_shop_first_sale', 'item_first_sale',
             #'date_shop_type_avg_item_cnt_lag_1','date_shop_subtype_avg_item_cnt_lag_1',
             'date_city_avg_item_cnt_lag_1', 'date_item_city_avg_item_cnt_lag_1',
             #'date_type_avg_item_cnt_lag_1', #'date_subtype_avg_item_cnt_lag_1',
             'delta_price_lag', 'month', 'days', 'item_shop_last_sale', 'item_last_sale']]
X_train = data[data.date_block_num < 34].drop(['item_cnt_month'], axis=1)
#Y_train = train_set['item_cnt']
y_train = data[data.date_block_num < 34]['item_cnt_month'].clip(0.,20.)

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

X_val = X_train[X_train.date_block_num > 30]
X_train = X_train[X_train.date_block_num <= 30]

y_val = y_train[~y_train.index.isin(X_train.index)]
y_train = y_train[y_train.index.isin(X_train.index)]

X_val_test = X_val[X_val.date_block_num > 32]
X_val = X_val[X_val.date_block_num <= 32]

y_val_test = y_val[~y_val.index.isin(X_val.index)]
y_val = y_val[y_val.index.isin(X_val.index)]

X_train.head()
def rmse(y_pred, y):
    return np.sqrt(np.mean(np.square(y - y_pred)))
lgtrain = lightgbm.Dataset(X_train, label=y_train)
lgval = lightgbm.Dataset(X_val, label=y_val)

def evaluate_metric(params):
    
    model_lgb = lightgbm.train(params, lgtrain, 600, 
                          valid_sets=[lgtrain, lgval], early_stopping_rounds=100, 
                          verbose_eval=300)

    pred = model_lgb.predict(X_val_test, num_iteration=1000)

    score = rmse(pred, y_val_test)
    
    print(score, params)
 
    return {
        'loss': score,
        'status': STATUS_OK,
        'stats_running': STATUS_RUNNING
    }
# Define searched space
hyper_space = {'objective': 'regression',
               'metric':'rmse',
               'boosting':'gbdt',
               #'n_estimators': hp.choice('n_estimators', [25, 40, 50, 75, 100, 250, 500]),
               'max_depth':  hp.choice('max_depth', [3, 5, 8, 10, 12, 15]),
               'num_leaves': hp.choice('num_leaves', [25, 50, 75, 100, 125, 150, 225, 250, 350, 400, 500]),
               'subsample': hp.choice('subsample', [.3, .5, .7, .8, .9, 1]),
               'colsample_bytree': hp.choice('colsample_bytree', [.5, .6, .7, .8, .9, 1]),
               'learning_rate': hp.choice('learning_rate', [.01, .1, .05, .2]),
               'reg_alpha': hp.choice('reg_alpha', [.1, .2, .3, .4, .5, .6, .7]),
               'reg_lambda':  hp.choice('reg_lambda', [.1, .2, .3, .4, .5, .6]), 
                # 'bagging_fraction': hp.choice('bagging_fraction', [.5, .6, .7, .8, .9, 1]),
               'feature_fraction':  hp.choice('feature_fraction', [.6, .7, .8, .9, 1]), 
               'bagging_frequency':  hp.choice('bagging_frequency', [.3, .4, .5, .6, .7, .8, .9]),                  
               'min_child_samples': hp.choice('min_child_samples', [10, 20, 30, 40])}
# Trial
trials = Trials()

# Set algoritm parameters
algo = partial(tpe.suggest, 
               n_startup_jobs=-1)

# Seting the number of Evals
MAX_EVALS= 30

# Fit Tree Parzen Estimator
best_vals = fmin(evaluate_metric, space= hyper_space, verbose=1,
                 algo=algo, max_evals=MAX_EVALS, trials=trials)

# Print best parameters


best_params = space_eval(hyper_space, best_vals)
print("BEST PARAMETERS: " + str(best_params))
model_lgb = lightgbm.train(best_params, lgtrain, 5000, 
                      valid_sets=[lgtrain, lgval], early_stopping_rounds=500, 
                      verbose_eval=250)

lgb_pred = model_lgb.predict(X_test).clip(0, 20)
lgb_pred = model_lgb.predict(X_test).clip(0, 20)

result = pd.DataFrame({
    "ID": test["ID"],
    "item_cnt_month": lgb_pred.clip(0. ,20.)
})
result.to_csv("submission.csv", index=False)

