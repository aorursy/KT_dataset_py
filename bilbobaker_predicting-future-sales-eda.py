import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import ceil

## Storing all the files into dataframes

sample = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
sales_test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
sales_test
sales_train.head()
items.head()
## Joining training data to the different item names
new_train = sales_train.join(items.set_index('item_id'),on='item_id')
## Joining training data to the different item catagories
new_train = new_train.join(categories.set_index('item_category_id'),on='item_category_id')
new_train = new_train.join(shops.set_index('shop_id'),on='shop_id')
new_train['item_price'].nunique()
new_train['item_name'].nunique()
## Need to figure out how to extract total sales by date

## This gives us total sales by shop
new_train.groupby('shop_name').sum()
new_train.groupby('date')
## Looking to see how many rows we have
new_train.shape
## Looking for outliers 
new_new = new_train[new_train['item_price']<=5000]
## Looking at the shape of the item prices without 
new_new['item_price'].hist()
outliers = new_train[new_train['item_price']>=5000]
outliers
33831/2935849
train_prepped = new_train[new_train['item_price']<=5000]
train_prepped
## Attempts at seeing if it clusters

print(train_prepped['shop_id'].nunique())
print('\b')
print(train_prepped['date_block_num'].nunique())

print(60*34)
train_prepped['date_time'] = pd.to_datetime(train_prepped['date'],format='%d.%m.%Y')
train_prepped
df= train_prepped.groupby(['shop_id','date_block_num'],as_index=False)['item_cnt_day'].sum()
df

df= train_prepped.groupby(['date_block_num','shop_id'],as_index=False)['item_cnt_day'].sum()
df

## Graphing Monthly Sales number by company
plt.figure(figsize=[10,18])

sns.lineplot(data=df,x='date_block_num',y='item_cnt_day',hue='shop_id',palette='dark')
plt.legend(bbox_to_anchor=(1.2,1))
monthly_sales=train_prepped.groupby(["date_block_num","shop_id","item_id"])[
    "date_time","item_price","item_cnt_day"].agg({"date_time":["min",'max'],"item_price":"mean","item_cnt_day":"sum"})
df
monthly_sales.head()
grouped = pd.DataFrame(train_prepped.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))
num_graph = 10
id_per_graph = ceil(grouped.shop_id.max() / num_graph)
count = 0
for i in range(5):
    for j in range(2):
        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])
        count += 1
len(train_prepped)
len(sales_test)
2902039/214200
sales_test['item_id'].nunique()
train_prepped['item_id'].nunique()


## Creating minimum date for each item
min_date = train_prepped.groupby(["item_id"]).agg({"date_time":"min"})
min_date

min_date['min_date'] = min_date['date_time']
min_date = min_date.drop('date_time',axis=1)
## Joining date to minimum date & adding column to training data
train_prepped = train_prepped.join(min_date,on='item_id')
train_prepped.head()
train_prepped['first_month'] = train_prepped.apply(first_month,axis=1)
train_prepped.head()
## train_prepped['first_month'] = (train_prepped['date_time'] > train_prepped['min_date']) & (train_prepped['date_time'] < (train_prepped['min_date'] + pd.Timedata('30 d'))
## Creating minimum date for each item at each shop
min_date_per_shop = train_prepped.groupby(["shop_id","item_id"]).agg({"date_time":"min"})
min_date_per_shop
## Creating indicator for first month sale at each shop
def first_month(row):
    if (row['date_time'] > min_date_per_shop.loc[(row['shop_id'],row['item_id']), 'date_time']) & (row['date_time'] < (min_date_per_shop.loc[(row['shop_id'],row['item_id']), 'date_time'] + pd.Timedelta('30 d'))):
        return 1
    else:
        return 0
min_date_per_shop.loc[(row['shop_id'],row['item_id']), 'date_time']
train_prepped['first_month'] = train_prepped.apply(first_month,axis=1)
aggregated_train = train_prepped.groupby(["shop_id","item_id"])["date_time","item_cnt_day",'first_month'].agg({"date_time":["min",'max'],"item_cnt_day":"sum",'first_month':'sum'})
aggregated_train
train_prepped.to_csv('training_prepped.csv')
train_prepped.head()
train_2 = train_prepped.groupby(['shop_id','item_id'])['item_cnt_day','first_month'].sum()
sales_test.head()
train_2.groupby('shop_id').mean()
