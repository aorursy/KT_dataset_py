import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))

import pandas as pd

data_test = pd.read_csv('../input/test.csv')
data_submition = pd.read_csv('../input/sample_submission.csv')

data_items = pd.read_csv('../input/items.csv')
data_shops = pd.read_csv('../input/shops.csv')
data_sales_train = pd.read_csv('../input/sales_train.csv')
data_item_categories = pd.read_csv('../input/item_categories.csv')


data_items.head()
data_shops.head()
data_item_categories.head()
data_sales_train.head()
# create prices

prices = data_sales_train.item_price.map(lambda x: str(x).split('.'))
data_sales_train['item_price_dollars'] = prices.map(lambda p : p[0]).astype('int32')
data_sales_train['item_price_cents'] = prices.map(lambda p : p[1]).astype('int32')

data_sales_train.head()


# number of sold items 
data_sales_train['item_sold_day'] = data_sales_train.item_cnt_day.astype('int32')

# data_sales_train.item_cnt_day.map(lambda x: 1 if x%1>0 else 0 ).sum()
data_sales_train.head()
# data_sales_train.groupby(['shop_id','item_id']).item_sold_month.sum()
#create dates

dates = data_sales_train.date.map(lambda x: str(x).split('.'))
data_sales_train['date_day'] = dates.map(lambda d : d[0]).astype('int32')
data_sales_train['date_month'] = dates.map(lambda d : d[1]).astype('int32')
data_sales_train['date_year'] = dates.map(lambda d : d[2]).astype('int32')
data_sales_train.head()
# DATE

# data_sales_train['date2'] = pd.to_datetime(data_sales_train.date, format='%m/%d/%Y')
pd.to_datetime(data_sales_train.date[0], format='%d.%m.%Y').weekday_name[0]
# data_sales_train.date[0].dt.weekday_name


df  = data_sales_train.sort_values(by=['shop_id','item_id','date_year','date_month','date_day'],ascending=True,inplace=False)
df.drop('date', axis=1, inplace=True)
df.drop('item_price', axis=1, inplace=True)
df.drop('item_cnt_day', axis=1, inplace=True)
df.drop(['date_day','date_month','date_year'], axis=1, inplace=True)
df.drop(['item_price_dollars','item_price_cents'], axis=1, inplace=True)
df.rename(columns={'item_sold_day':'item_sold_month'},inplace=True)
df.head()
d = df.loc[(df['shop_id'] == 59) & (df['item_id']== 30)]

for n in range(0,35):
    print(n)
# data_item_categories.item_category_name

d = data_item_categories.item_category_name.apply(lambda x: pd.value_counts(x.split(" "))).sum(axis = 0).sort_values(ascending=False)
d = d[1:]

