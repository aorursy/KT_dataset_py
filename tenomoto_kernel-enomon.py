# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import plotly
plotly.offline.init_notebook_mode(connected=False)
import cufflinks as cf
# csv read
item_cat = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')
sales_train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')
sample_sub = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')
item_cat
items
item_join = pd.merge(items, item_cat[['item_category_name','item_category_id']], on='item_category_id', how='left')
item_join.head()
sales_train
join_data = pd.merge(sales_train, item_join[['item_name','item_id','item_category_name','item_category_id']], on='item_id', how='left')
join_data.head()
sample_sub
shops
master = pd.merge(join_data, shops[['shop_name','shop_id']], on='shop_id', how='left')
master.head()
test['shop_id'].nunique()
test
master.head()
master['item_id'].nunique()
master = master.sort_index(axis=1, ascending=True)
master['itemtotal'] = master['item_cnt_day'] * master['item_price']
master.head()
master.head()
master.info()
master.nunique()
master_multi = master.set_index(['shop_id','item_category_id','date_block_num','date'])
master_multi.head()
print(master_multi.sum(level='date_block_num'))

master_multi.sum(level='date').plot.line(y='itemtotal')
master_multi.sum(level='date_block_num').plot.line(y='item_cnt_day')
master_multi.sum(level='date_block_num').plot.line(y='itemtotal')
# 返却入れる
master_multi.sum(level='date').plot.line(y='item_price')
master_multi.sum(level='date').iplot().line(y='item_price')
master_multi.max()
sns.barplot(x='item_id',y='item_price',data=master_multi)
# IDごとの予測を立てないといけない