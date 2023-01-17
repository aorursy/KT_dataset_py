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
item_categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
item_categories
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_sub = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
items
sample_sub
sales_train
shops.head()
test
sales_join = pd.merge(sales_train, items, on="item_id", how="left")
sales_join.head()
sales_join = pd.merge(sales_join,item_categories,on="item_category_id",how="left")
sales_join.head()
sales_join = pd.merge(sales_join,shops,on="shop_id",how="left")
sales_join.head()
#ショップごとの売り上げ総額を確認

shop_sales = sales_join.groupby("shop_name").sum()["item_price"]
shop_sales.plot.bar(figsize = (15, 8), width = 1)
print(sales_join.dtypes) 
shop_salescount = sales_join.groupby("shop_name").count()["date"]
shop_salescount.plot.bar(figsize = (15, 8), width = 1)
import datetime as dt
sales_join['date'] = pd.to_datetime(sales_join['date'])
sales_join.head()
sales_join["month"] = sales_join["date"].dt.strftime("%Y%m")
sales_join.head()
shop_sale_month = sales_join.groupby(["month"]).count()["item_cnt_day"]
shop_sale_month.plot.bar(figsize = (15, 8), width = 1)
sample_sub.head()
test_join = pd.merge(test,items,on="item_id",how="left")
test_join.head()
test_join = pd.merge(test_join,item_categories,on="item_category_id",how="left")
test_join.head()
test_join = pd.merge(test_join,shops,on="shop_id",how="left")
test_join.head()
item_sale_month = sales_join.pivot_table(index="month",columns="item_category_name",aggfunc="size",fill_value=0)
item_sale_month
item_sale_month.plot(figsize=(18,5))
