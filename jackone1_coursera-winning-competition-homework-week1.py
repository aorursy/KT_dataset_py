# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input/"))

# Any results you write to the current directory are saved as output.
transactions    = pd.read_csv('../input/sales_train.csv.gz')
items           = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops           = pd.read_csv('../input/shops.csv')
print("transactions:")
print(transactions.shape)
transactions.head()
print("items:")
print(items.shape)
items.head()
print("item_categories:")
print(item_categories.shape)
item_categories.head()
print("shops:")
print(shops.shape)
shops.head()
transactions.dtypes
transactions["date"] = pd.to_datetime(transactions.date, format="%d.%m.%Y")
transactions.dtypes
date_range = pd.date_range(start="2014-09-01",end="2014-09-30")
sep_transactions = transactions[transactions['date'].isin(date_range)]
sep_transactions.head()

# or
# sep_transactions = transactions[transactions['date'].dt.year==2014]
# sep_transactions = sep_transactions[sep_transactions['date'].dt.month==9 ]
# sep_transactions.head()
# get the data that item_cnt_day>0
# sep_transactions =  sep_transactions[sep_transactions['item_cnt_day']>=0]
# create a new column to save the value that item_price multiply item_cnt_day
sep_transactions['sale'] = sep_transactions.item_price*sep_transactions.item_cnt_day
# group by shop_id
each_shop_sales = sep_transactions['sale'].groupby(sep_transactions['shop_id']).sum()
# get the max value
print("shop_id:"+str(each_shop_sales.idxmax()) )
print("value:"+str(each_shop_sales.max()) )
# the summer is from June 21 to September 23 each year
date_range = pd.date_range(start="2014-06-21",end="2014-09-23")
summer_data = transactions[transactions['date'].isin(date_range)]
summer_data.head()
# merge the summer_data dataframe and items dataframe to create a new column called item_category_id
new_summer_data =pd.merge(summer_data,items,how="inner",on="item_id")
new_summer_data.head()
# get the data that item_cnt_day>0
# new_summer_data =  new_summer_data[new_summer_data['item_cnt_day']>0]

# create a new column to save the value that item_price multiply item_cnt_day
new_summer_data['sale'] = new_summer_data.item_price*new_summer_data.item_cnt_day
# group by shop_id
each_category_sale = new_summer_data['sale'].groupby(new_summer_data['item_category_id']).sum()
# get the max value
print("item_category_id:"+str(each_category_sale.idxmax()) )
print("value:"+str(each_category_sale.max()) )
groups = transactions['item_price'].groupby(transactions['item_id'])
# groups.describe()
item_ids=[]
for item_id,group in groups:
    if len(group.unique())==1:
        item_ids.append(item_id)
# print(item_ids)
print(len(item_ids))
date_range = pd.date_range(start="2014-12-01",end="2014-12-31")
shop_25 = transactions[transactions['shop_id']==25]
shop_25 = shop_25[shop_25['date'].isin(date_range)]
# i think the row that item_cnt_day < 0 are outlier,so we can remove this data before we 
# shop_25 = shop_25[shop_25['item_cnt_day']>=0]
groups = shop_25['item_cnt_day'].groupby(shop_25['date'])
names=[]
nums = []
for name,group in groups:
    names.append(name)
    nums.append(group.sum())
print(nums)
pd.Series(groups.sum()).var()
# pd.Series(nums).var()
