# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

items_df = pd.read_csv('../input/items.csv')
shops_df = pd.read_csv('../input/shops.csv')

icats_df = pd.read_csv('../input/item_categories.csv')
train_df = pd.read_csv('../input/sales_train.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
smpsb_df = pd.read_csv('../input/sample_submission.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
test_df  = pd.read_csv('../input/test.csv.gz', compression='gzip', header=0, sep=',', quotechar='"')
# Any results you write to the current directory are saved as output.
l = list(icats_df.item_category_name)
l_cat = l

for ind in range(0,8):
    l_cat[ind] = 'Access'

for ind in range(10,18):
    l_cat[ind] = 'Consoles'

for ind in range(18,25):
    l_cat[ind] = 'Consoles Games'

for ind in range(26,28):
    l_cat[ind] = 'phone games'

for ind in range(28,32):
    l_cat[ind] = 'CD games'

for ind in range(32,37):
    l_cat[ind] = 'Card'

for ind in range(37,43):
    l_cat[ind] = 'Movie'

for ind in range(43,55):
    l_cat[ind] = 'Books'

for ind in range(55,61):
    l_cat[ind] = 'Music'

for ind in range(61,73):
    l_cat[ind] = 'Gifts'

for ind in range(73,79):
    l_cat[ind] = 'Soft'


icats_df['cats'] = l_cat

#google translate
icats_df['cats'] = icats_df['cats'].replace('Чистые носители (штучные)','Clean media (piece)')
icats_df['cats'] = icats_df['cats'].replace('Чистые носители (шпиль)','Clean carriers (spire)')
icats_df['cats'] = icats_df['cats'].replace('Служебные - Билеты','Official - Tickets')
icats_df['cats'] = icats_df['cats'].replace('Игры - Аксессуары для игр', 'Games')
icats_df['cats'] = icats_df['cats'].replace('Доставка товара','Delivery of goods')
icats_df['cats'] = icats_df['cats'].replace('Билеты (Цифра)','Tickets (figure)')
icats_df['cats'] = icats_df['cats'].replace('PC - Гарнитуры/Наушники', 'Headphones')
icats_df['cats'] = icats_df['cats'].replace('Служебные','Office')



icats_df.head()
joined = pd.merge(train_df, items_df, left_on='item_id', right_on='item_id')
joined = pd.merge(joined, icats_df, left_on='item_category_id', right_on='item_category_id')
joined.head()
joined['date'] = pd.to_datetime(joined['date'], errors='coerce')
joined['year'] = joined['date'].dt.year
joined['month'] = joined['date'].dt.month
joined['day'] = joined['date'].dt.day
import matplotlib.pyplot as plt

allCats = joined['cats'].unique()
print(allCats)
for cat in allCats:
    data = joined.loc[joined['cats'] == cat].groupby(["year", "month"]).agg({"year": "count"})
    data.plot(kind="bar", title=cat)
    
returned = joined.loc[joined['item_cnt_day'] <0].groupby(["year", "month"]).agg({"year": "count"})
data.plot(kind="bar", title="returned items")
import seaborn as sns
corr = joined.drop(['date_block_num'], axis=1).corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)