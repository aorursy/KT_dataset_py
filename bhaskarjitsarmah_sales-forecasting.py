import numpy as np
import pandas as pd
items = pd.read_csv('../input/items.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
shops = pd.read_csv('../input/shops.csv')
sales_train = pd.read_csv('../input/sales_train.csv.gz', parse_dates=['date'])
items.head()
items['item_name'].value_counts()
items['item_category_id'].value_counts()
item_categories.head()
item_with_categories = pd.merge(items, item_categories, on='item_category_id')
item_with_categories.head()
shops.head()
sales_train.head()
shops_with_sales = pd.merge(shops, sales_train, on='shop_id')
shops_with_sales.head()
train = pd.merge(shops_with_sales, item_with_categories, on='item_id')
train.head()
train['sales'] = train['item_price'] * train['item_cnt_day']
train.head()
train['shop_id'].value_counts()
train['item_id'].value_counts()
%matplotlib inline
import matplotlib.pyplot as plt

def plot_data(train, start_date, end_date, item_id, shop_id, column='sales'):
    train = train.set_index('date')
    train = train.loc[start_date:end_date]
    train = train[train['item_id'] == item_id]
    train = train[train['shop_id'] == shop_id]
    train.plot(y='sales')
    plt.show()
train.head()
plot_data(train, '2012-12-01', '2015-12-01', 20949, 31)
plot_data(train, '2013-12-01', '2014-04-01', 20949, 31)
