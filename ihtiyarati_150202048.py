# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sqlite3

from datetime import datetime
from numpy import concatenate
from math import ceil
from math import sqrt

%matplotlib inline
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
db = sqlite3.connect("final.db")
train.to_sql('train', con=db, if_exists='replace')
del train

test.to_sql('test', con=db, if_exists='replace')
del test

items.to_sql('items', con=db, if_exists='replace')
del items
#sql sorgusu
sql_string = ("SELECT train.date, train.date_block_num, train.shop_id, train.item_id, train.item_price, train.item_cnt_day " + 
              "FROM train, test " + 
              "WHERE train.shop_id = test.shop_id AND train.item_id = test.item_id " + 
              "GROUP BY train.shop_id, train.item_id, train.date_block_num")
train = pd.read_sql(sql_string, db)
train.head()
#veride null değer var mı?
train.isnull().any()
#date sütununu formatla
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')

#date sütunundaki değerleri ayır
train['day'] = train['date'].dt.day
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year

train.head()
#verileri veritabanına yeni bir tablo olarak kaydet
train.to_sql('train_clear', con=db, if_exists='replace')
#grafiklere grid ekleme
sns.set(style="whitegrid")
#subplot satır, sütun sayısının belirlenmesi
row = 5
col = 2
#grafik sayısının hesaplanması
number_of_graph = row * col

#verinin shop_id sütununa göre gruplanması
group_by_shop = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())
#grafik başına düşen mağaza sayısının bulunması
store_per_graph = ceil(group_by_shop.shop_id.max() / number_of_graph)

#grafiklerin genel kurallarını belirleme
fig, axes = plt.subplots(nrows=row, ncols=col, sharex=True, sharey=True, figsize=(18,20))
#verilerin grafiklere eklenmesi ve grafiğin ekrana basılması
count = 0
for i in range(5):
    for j in range(2):
        #sıradaki satırların shop_id bazında seçilmesi
        data=group_by_shop.loc[(group_by_shop['shop_id'] >= count*store_per_graph) & (group_by_shop['shop_id'] < (count+1)*store_per_graph)]
        #grafiğin basılması
        sns.lineplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=data, ax=axes[i][j], palette="tab10")
        count += 1
#sql sorgusu
sql_string = ("SELECT train_clear.date, train_clear.date_block_num, train_clear.shop_id, train_clear.item_id, train_clear.month, train_clear.item_cnt_day, items.item_category_id " + 
              "FROM train_clear, items " + 
              "WHERE train_clear.item_id = items.item_id ")
train = pd.read_sql(sql_string, db)
train.head()
#veride null değer var mı?
train.isnull().any()
#grafiklere grid ekleme
sns.set(style="whitegrid")
#subplot satır, sütun sayısının belirlenmesi
row = 5
col = 2
#grafik sayısının hesaplanması
number_of_graph = row * col

#verinin item_category_id sütununa göre gruplanması
group_by_categories = pd.DataFrame(train.groupby(['item_category_id', 'month'])['item_cnt_day'].sum().reset_index())
#grafik başına düşen mağaza sayısının bulunması
items_per_graph = ceil(group_by_categories.item_category_id.max() / number_of_graph)

#grafiklerin genel kurallarını belirleme
fig, axes = plt.subplots(nrows=row, ncols=col, sharex=True, sharey=True, figsize=(18,20))

#verilerin grafiklere eklenmesi ve grafiğin ekrana basılması
count = 0
for i in range(5):
    for j in range(2):
        #sıradaki satırların belirli bir item_category_id aralığında seçilmesi
        data=group_by_categories.loc[(group_by_categories['item_category_id'] >= count*items_per_graph) & (group_by_categories['item_category_id'] < (count+1)*items_per_graph)]
        #grafiğin basılması
        sns.lineplot(x='month', y='item_cnt_day', hue='item_category_id', data=data, ax=axes[i][j], palette="tab10")
        count += 1
