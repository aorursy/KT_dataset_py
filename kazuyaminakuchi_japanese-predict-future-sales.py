# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

sales_train
# おかしな型はなさそう

sales_train.dtypes
# 欠損確認

# 欠損なし

sales_train.isnull().sum()
# 統計量確認

# - item_price:最小値マイナスは異常では？。最大値307,980は大きすぎないか？

# - item_cnt_day:最小値マイナスは異常では？。最大値2,169は大きすぎないか？

sales_train.describe().apply(lambda x: round(x, 2))
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

test
# おかしな型はなさそう

test.dtypes
# 欠損なし

test.isnull().sum()
test.describe().apply(lambda x: round(x, 2))
sample_submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')

sample_submission
sample_submission.dtypes
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

items
items.dtypes
items.isnull().sum()
items.describe().apply(lambda x: round(x, 2))
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

item_categories
item_categories.dtypes
item_categories.isnull().sum()
item_categories.describe().apply(lambda x: round(x, 2))
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

shops
shops.dtypes
shops.isnull().sum()
shops.describe().apply(lambda x: round(x, 2))
# 商品情報を結合する

sales_train_merged = (

    sales_train

    .join(items.set_index('item_id'), on='item_id')

    .join(item_categories.set_index('item_category_id'), on='item_category_id')

)

sales_train_merged
# 日付型に変換

sales_train_merged['date'] = pd.to_datetime(sales_train_merged['date'], format='%d.%m.%Y')
# 年列

sales_train_merged['year'] = sales_train_merged['date'].dt.year
# 月列

sales_train_merged['month'] = sales_train_merged['date'].dt.month
# 日列

sales_train_merged['day'] = sales_train_merged['date'].dt.day
# 曜日列追加(数字)

# Monday:0

# Tuesday:1

# Wednesday:2

# Thursday:3

# Friday:4

# Saturday:5

# Sunday:6

sales_train_merged['day_of_the_week_num'] = sales_train_merged['date'].dt.dayofweek
# 並べ替え

sales_train_merged.sort_values(by=['date', 'shop_id', 'item_id'], inplace=True)
# インデックス振り直し

sales_train_merged.reset_index(inplace=True, drop=True)

sales_train_merged
# 月ごとのデータ数分布

# 全体的に減少傾向

# 2つ飛び出てる月がある

sales_train_merged['date_block_num'].hist(bins=34)
# 飛び出てるのは11, 23 = 2013-12, 2014-12

# クリスマスがあるから売上が上がっている?

sales_train_merged['date_block_num'].value_counts()
# 各年のデータ数分布

sales_train_merged['year'].hist(bins=3)
# 各月のデータ数分布

# 2015年は11, 12月のデータが無いので注意

sales_train_merged['month'].hist(bins=12)
# 各日のデータ数分布

# 29-31日は無い月があるので注意

# 大きな偏りはない

sales_train_merged['day'].hist(bins=31)
# 曜日毎のデータ数

# 週末が多い

# 特に5=土曜が多い

sales_train_merged['day_of_the_week_num'].hist(bins=7)
# shop毎のデータ数

# 偏りあり

sales_train_merged['shop_id'].hist(bins=60)
# 商品毎のデータ数

# item_id=20949が飛び抜けて多い

sales_train_merged['item_id'].value_counts()
# 商品名「コーポレートパッケージTシャツ1Cインタレストホワイト」(google翻訳で日本語にしたもの)

sales_train_merged[sales_train_merged['item_id'] == 20949].head(1)
# 商品毎のデータ数

# item_id=20949以外は、そこそこ均等

sales_train_merged['item_id'].hist(bins=range(0, 32501, 2500))
# 商品カテゴリ毎のデータ数

# 偏りあり

sales_train_merged['item_category_id'].hist(bins=84)
# date_block_num毎

# やはり11, 23 = 2013-12, 2014-12が多い

sales_train_merged.groupby('date_block_num')['item_cnt_day'].sum().plot(kind='bar')
# 年

# 減少傾向

sales_train_merged.groupby('year')['item_cnt_day'].sum().plot(kind='bar')
# 月

# ※2015年は11, 12月のデータなし

# その割に12月は多い

sales_train_merged.groupby('month')['item_cnt_day'].sum().plot(kind='bar')
# 曜日

# 週末が多く、土曜が一番多い

sales_train_merged.groupby('day_of_the_week_num')['item_cnt_day'].sum().plot(kind='bar')
# 日

# 31日は存在しない月が多いので不利

# 大きな偏りはなさそう

sales_train_merged.groupby('day')['item_cnt_day'].sum().plot(kind='bar')
# ショップ毎

# ショップ間で差が激しい

# sales_train_merged.groupby('shop_id')['item_cnt_day'].sum().plot(kind='bar', figsize=(15, 4))

sales_train_merged.groupby('shop_id')['item_cnt_day'].sum().sort_values(ascending=False).plot(kind='bar', figsize=(15, 4))
# 商品カテゴリ毎

# カテゴリ間で差が大きい

sales_train_merged.groupby('item_category_id')['item_cnt_day'].sum().sort_values(ascending=False).plot(kind='bar', figsize=(15, 4))
# 商品毎

# item_id=20949だけ一桁多い

# マイナスがある

sales_train_merged.groupby('item_id')['item_cnt_day'].sum().sort_values(ascending=False)
sales_train_merged.head()