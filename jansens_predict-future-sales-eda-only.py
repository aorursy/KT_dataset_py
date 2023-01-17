import numpy as np

from numpy import array

import pandas as pd

from pandas import read_csv, Series, DataFrame, to_datetime



import pickle



import warnings

warnings.filterwarnings('ignore')
import sklearn as sklearn

import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor
import matplotlib as mpl

import matplotlib.pyplot as plt

from matplotlib.pyplot import pie, plot, scatter, show, title, xlabel, ylabel, xticks

%matplotlib inline 
pd.set_option('display.max_rows', 600)

pd.set_option('display.max_columns', 50)

pd.plotting.backend='hvplot'
import seaborn as sns

sns.set(rc={'figure.figsize':(20, 5)})

from seaborn import set_context, barplot, boxplot
from palettable.colorbrewer.qualitative import Pastel1_7, Pastel1_8, Pastel1_9
from plotly.graph_objects import Figure, Pie
!pip install googletrans
for p in [mpl, np, pd, sklearn, lgb, sns]:

    print (p.__name__, p.__version__)
def translate_to_english(word):



    from googletrans import Translator

    translator = Translator()

    translated = translator.translate(word)

    

    return translated.text
raw_dir = '../input/competitive-data-science-predict-future-sales/' 

sales = read_csv(raw_dir+'sales_train.csv')

shops = read_csv(raw_dir+'shops.csv')

items = read_csv(raw_dir+'items.csv')

item_cats = read_csv(raw_dir+'item_categories.csv')

test = read_csv(raw_dir+'test.csv')
sales.sample(8).sort_index()
from datetime import datetime as dt

try:

    sales.date = sales.date.apply(lambda x: dt.strptime(x, '%d.%m.%Y'))

except:

    pass
sales['year_month'] = to_datetime(sales['date']).apply(lambda x: '{year}-{month}'.format(year=x.year, month= '{:02d}'.format(x.month)))
sales.sample(8).sort_index().sort_values('date_block_num')
sales.date.min(), sales.date.max()
time_range = sales.date.max() - sales.date.min() 

time_range
sales.describe()
sales[sales.item_price < 0]
sales = sales[sales.item_price > 0]
sales[sales.item_price == sales.item_price.max()]
boxplot(x=sales.item_price)

show()
sales = sales[(sales.item_price < 307980.0) & (sales.item_price > 0)]
boxplot(x=sales.item_price)

show()
sales[sales.item_cnt_day < 0]
sales.item_cnt_day.mask(sales.item_cnt_day <0, 0, inplace=True)
boxplot(x=sales.item_cnt_day)

show()
sales.item_cnt_day.max()
sales = sales[sales["item_cnt_day"] < 2000]
boxplot(x=sales.item_cnt_day)

show()
len(sales['shop_id'].unique())
unique_shops = array(sorted(sales['shop_id'].unique()))

unique_shops, len(unique_shops)
pie(unique_shops,

    labels=unique_shops,

    labeldistance=1.0,

    colors=Pastel1_9.hex_colors,

    textprops={'fontsize': 8},

    rotatelabels = True,

    startangle=-90

   )

title('60 SHOP IDs')

show()
_ = DataFrame(sales.groupby('date_block_num')['shop_id'].nunique())

_.sample(8).sort_index()
set_context("talk", font_scale=1.1)

barplot(

    data = _,

    x = _.index,

    y = _.shop_id

);

plot(_.shop_id)



title('\nNUMBER SHOPS SELLING per DATE BLOCK\n')

xlabel('\nDATE BLOCK')

ylabel('n of SHOPs-SELLING\n')

xticks(rotation=75, fontsize='xx-small')



_.plot()



del _

show()
_ = DataFrame(sales.groupby(['date_block_num']).sum().item_cnt_day).reset_index()

_.head(8)
set_context("talk", font_scale=1.0)

barplot(

    data = _,

    x = 'date_block_num', 

    y = 'item_cnt_day'

);

plot(_.item_cnt_day)

title('\nNUMBER OF ITEMS SOLD per DATE BLOCK\n')

xlabel('\nDATE BLOCK')

ylabel('ITEMS SOLD\n')

xticks(rotation=75, fontsize='x-small' )

_.plot()

title('\nNUMBER OF ITEMS SOLD per DATE BLOCK\n')

xlabel('\nDATE BLOCK')

ylabel('ITEMS SOLD\n')

xticks(rotation=75, fontsize='x-small' )

del _

show()
sales["weekday"] = sales.date.apply(lambda x: x.weekday())

sales.groupby("date")["item_cnt_day"].sum().plot(figsize=(15,7))

show()
_ = DataFrame(sales.groupby("weekday")["item_cnt_day"].sum().sort_index())

week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

_.index = _.index.to_series().apply(lambda x: week[x])

_
title('\nSALES PER WEEKDAY\n')

xlabel('\nWeekday')

ylabel('Items Sold\n')

xticks(rotation=75, fontsize='x-small' )



barplot(

    data = _,

    x = _.index,

    y = _.item_cnt_day

)

_.plot()



title('\nSALES PER WEEKDAY\n')

xticks(rotation=75, fontsize='x-small' )

#xlabel('\nWeekday')

ylabel('Items Sold\n')



show()
pie(_.item_cnt_day,

    labels=_.index, 

    colors=Pastel1_7.hex_colors)

del _

show()
_ = DataFrame(sales.groupby('year_month')['item_cnt_day'].sum())

_
title('\nSALES PER MONTH\n')

xticks(rotation=75, fontsize='x-small' )

xlabel('\nMONTH of YEAR')

ylabel('Items Sold\n')



barplot(

    data = _,

    x = _.index,

    y = _.item_cnt_day

       )

plot(_.item_cnt_day)



_.plot()



title('\nSALES PER MONTH\n')

xticks(rotation=75, fontsize='x-small' )

xlabel('\nMONTH of YEAR')

ylabel('Items Sold\n')

del _

show()
_ = DataFrame(sales.groupby(['shop_id']).sum()['item_cnt_day'])

_ = _.sort_values('item_cnt_day', ascending=False)

_.head(8)
set_context("talk", font_scale=1.1)

chart = barplot(

    data = _,

    x = _.index, 

    y = _.item_cnt_day

);



title('\nITEMS SOLD per SHOP IDs\n')

xlabel('\nSHOP ID #')

ylabel('ITEMS SOLD\n')

xticks(rotation=60, fontsize='xx-small' )



_.sort_index().plot()

title('\nITEMS SOLD per SHOP IDs\n')

xlabel('\nSHOP ID #')

ylabel('ITEMS SOLD\n')

del _

xticks(rotation=60, fontsize='xx-small' )

show()
_ = DataFrame(sales.groupby(['item_id']).sum()['item_cnt_day'])

_ = _.sort_values('item_cnt_day', ascending=False)

_.head(8)
set_context("talk", font_scale=1.4)

_.sort_index().plot();

xlabel('item id');

ylabel('sales');

show()
max_item_id = _['item_cnt_day'].idxmax()

max_item_id
_ = _[_.index != max_item_id]

sales = sales[sales.item_id !=max_item_id]

del max_item_id
_.sort_index().plot()

title('ITEMS SOLD FOR EACH ITEM ID\n')

xlabel('\nITEM ID#')

ylabel('ITEMS SOLD\n')

show()
sales.groupby(['item_id']).sum().sort_values(['item_cnt_day'], ascending=False).head(10)[['item_cnt_day']]
_ = sales.groupby(['item_id']).sum()

_ = _.sort_values(['item_cnt_day'], ascending=True)[['item_cnt_day']]

_[_.item_cnt_day==1.0]
sales.date.max()
DataFrame(sales[sales.date == sales.date.max()]['item_id'].unique()).head(10)
shops.sample(8).sort_index()
unique_shops = shops['shop_id'].unique()

unique_shop_names = shops['shop_name'].unique()



if len(unique_shops) != len(unique_shop_names):

    print("There are different shop ids with the same name!")

else:

    print("There are "+str(len(unique_shops))+" unique shops and none of them have the same name!")
shop_name_lengths = {}

for r in shops['shop_name']:

    shop_name_lengths[r] = len(r)

shop_names = Series(shop_name_lengths).sort_values()
shop_names = DataFrame(shop_names.reset_index())

shop_names.columns=['name','name_length']

shop_names.sample(8).sort_index()
set_context('talk', font_scale=1.2)

data = shop_names.groupby(['name_length']).count()

barplot(data=data, x=data.index, y=data.name)

title('\nNUMBER OF SHOPS\nwith CERTAIN NAME LENGTH\n')

xlabel('\nNAME LENGTH (number of characters)\n')

ylabel('SHOPS\n')

show()
shop_names.hist(bins=shop_names['name_length'].nunique()+1)

title('NUMBER OF SHOPS\nwith CERTAIN NAME LENGTH\n')

xlabel('\nNAME LENGTH (number of characters)\n')

ylabel('Number of Shops\n')

show()
_ = shop_names.iloc[shop_names['name_length'].idxmax()]

length = int(_.name_length)

print("The longest shop name is:\n"+ _['name'] + "\n" +str(length) + " characters long")

del _
shops.shop_name.str.split().apply(lambda x: x[0]).unique()
shops['city'] = shops.shop_name.str.split().apply(lambda x: x[0].strip())

shops['city_en'] = shops.city.apply(lambda w: translate_to_english(w))
shops.sample(8).sort_index()
shops['shop_name_en'] = shops.shop_name.apply(lambda w: translate_to_english(w))

shops.head(10)
shops.city_en.nunique()
shops.groupby('city_en').count().sort_values('shop_id', ascending=False)
_ = shops.groupby('city_en').count()

_ = _.sort_values('shop_id', ascending=False).reset_index()

_ = _.groupby('shop_id').count()[['shop_name']]
_ = sales.merge(shops, how='left', on='shop_id')

_ = _.groupby('city_en').sum()[['item_cnt_day']]

_ = _.sort_values('item_cnt_day', ascending=False)

_.head(10)
items.sample(8).sort_index()
items.item_category_id.nunique()
n_items = items.item_id.nunique()

n_items
sales.date.max()
_ = sales[sales.date == sales.date.max()][['item_id']]

_ = _.sort_values('item_id').groupby('item_id').count()

_
items[items.item_id.isin(_.index.to_list())]
len(_)
percentage = len(_)/n_items * 100

print('percentage: ' + str('{:.3}'.format(percentage)) + " % items were sold recently.")
sales.date.min()
sales[sales.date == sales.date.min()][['item_price','item_id']].sort_values(['item_price'], ascending=False).head(10)
_ = sales[sales.date == sales.date.min()][['item_price','item_id']]

_ = _.sort_values(['item_price'], ascending=False)

_ = _.head(10)['item_id'].to_list()

_
items[items.item_id.isin(_)][['item_name']]
sales.groupby(['item_id']).sum().sort_values(['item_cnt_day'], ascending=False).head(10)[['item_cnt_day']]
_ = sales.groupby(['item_id']).sum()[['item_cnt_day']]

_ = _.sort_values(['item_cnt_day'], ascending=False)

_ = _.head(10).index.to_list()

_
items[items.item_id.isin(_)][['item_name']]
sales.date.min()
sales[sales.date == sales.date.min()].groupby(['item_id']).sum()[['item_cnt_day']].sort_values(['item_cnt_day'], ascending=False).head(10)
_ = sales[sales.date == sales.date.min()]

_ = _.groupby(['item_id']).sum()[['item_cnt_day']]

_ = _.sort_values(['item_cnt_day'], ascending=False)

_ = _.head(10).index.to_list()

_
items[items.item_id.isin(_)][['item_name']]
item_cats.sample(8).sort_index()
item_cats['item_category_name_en'] = item_cats['item_category_name'].apply(lambda w: translate_to_english(w))

item_cats.head(10)
item_cats.item_category_id.nunique()
item_cats.item_category_name.nunique()
_ = sales.merge(items,how='left', on='item_id')

_ = _.groupby('item_category_id').item_cnt_day.sum()

_ = DataFrame(_)

_.sample(8).sort_index()
set_context("talk", font_scale=1.5)



barplot(

    data = _,

    x = _.index, 

    y = 'item_cnt_day'

);



title('\nITEM CATEGORY SOLD\n')

xlabel('\nItem Category ID')

ylabel('Items Sold\n')

xticks(rotation=85, fontsize='xx-small' )

show()



_.plot()



title('\nITEM CATEGORY SOLD\n')

xlabel('\nItem Category ID')

ylabel('Items Sold\n')

xticks(rotation=85, fontsize='xx-small' )



del _

show()
item_cats.sample(5).sort_index()
cat_split = item_cats.item_category_name.str.split(" - ")



item_cats["item_group"] = cat_split.apply(lambda x:  x[0].strip())

item_cats['item_group_en'] = item_cats['item_group'].apply(lambda w: translate_to_english(w))



item_cats["item_subgroup"] = cat_split.apply(lambda x:  x[1].strip() if len(x) == 2 else x[0].strip())

item_cats['item_subgroup_en'] = item_cats['item_subgroup'].apply(lambda w: translate_to_english(w))



item_cats.sample(8).sort_index()[['item_category_id', 'item_category_name_en','item_group_en','item_subgroup_en']]
item_cats["item_group"].nunique()
set(item_cats["item_group_en"].values)
item_cats["item_subgroup"].nunique()
all_train_data = sales.merge(items, how='left', on='item_id')

all_train_data = all_train_data.merge(item_cats, how='left', on='item_category_id')

all_train_data = all_train_data.merge(shops, how='left', on='shop_id')

all_train_data.head(1)
!mkdir ../working/processed
necessary_columns = ['date','date_block_num','shop_id','item_id','item_price','item_cnt_day','year_month','weekday','item_category_id','item_group_en','item_subgroup_en','city_en']

all_train_data[necessary_columns].to_csv('../working/processed/all_train_data.csv.gz', compression='gzip')

!ls -al ../working/processed/
_ = all_train_data.groupby('item_group_en').sum()[['item_cnt_day']]

_ = _.sort_values('item_cnt_day', ascending=False)

_.head(10)
_ = all_train_data[all_train_data.item_group_en =='Movie']

_ = _.groupby('shop_id').sum()[['item_cnt_day']]

_ = _.sort_values('item_cnt_day', ascending=False).head(10)

_
_ = all_train_data.groupby('city_en')[['item_category_id']].nunique()

_ = _.sort_values('item_category_id', ascending=False)

_
_ = all_train_data.groupby('city_en')[['item_group']].nunique()

_ = _.sort_values('item_group', ascending=False)

_
def func(val, _):

    return _.item_group.ix[val]
labels = _.index

values = _.item_group

fig = Figure(data=[Pie(labels=labels, values=values, hole=.6)])

fig.update_layout(

    title={

        'text': "Number of Item Groups for Each City",

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.update_traces(hoverinfo='label+value', textinfo='label+value', textfont_size=8)

fig.show()
groups = item_cats['item_group_en'].unique()

cities = shops['city_en'].unique()

print(len(groups), len(cities))
_ = all_train_data.groupby(['city_en','item_group_en']).count()[['item_cnt_day']]

_ = DataFrame(_.index.to_list(), columns =['city_en', 'item_group_en'])

_.sample(10).sort_index()
_ = _.groupby('item_group_en').agg({'city_en':'count'}).sort_values('city_en', ascending = False)

_
labels = _.index

values = _.city_en

fig = Figure(data=[Pie(labels=labels, values=values, hole=.6)])

fig.update_layout(

    title={

        'text': "Number of City<br>for Each Item Groups",

        'y':0.9,

        'x':0.3,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.update_traces(hoverinfo='label+value', textinfo='label+value', textfont_size=8)

fig.show()
_[_.city_en == 32]
_[_.city_en == 1]
for city in all_train_data[all_train_data.item_group_en.isin(_[_.city_en == 1].index.to_list())]['city_en'].unique():

    print(city)
_ = all_train_data[['year_month','item_group_en','item_cnt_day']]

_ = _.groupby(['year_month','item_group_en']).sum()

_.head(17)
plot(_.item_cnt_day.unstack())

#plt.legend(all_train_data.item_group_en.unique())

title('\nSALES for EACH ITEM GROUP\n')

xlabel('\nDATE BLOCK')

ylabel('SALES\n')

xticks(rotation=75, fontsize='xx-small')

show()