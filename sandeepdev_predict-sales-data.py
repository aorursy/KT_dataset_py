# importing all the necessary modules required considering they are preinstalled.



import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

from itertools import product

from sklearn.preprocessing import LabelEncoder



# data sets available in ../input directory



import os

print(os.listdir("../input"))



# listing all the files in the input directory
# reading all the files

items = pd.read_csv('../input/items.csv')

shops = pd.read_csv('../input/shops.csv')

cats = pd.read_csv('../input/item_categories.csv')

train = pd.read_csv('../input/sales_train.csv')

test  = pd.read_csv('../input/test.csv')
train.item_price.describe()
plt.figure()

train.item_cnt_day.plot(kind='box')
plt.figure()

plt.xlim(train.item_price.min(), train.item_price.max()*1,1)

train.item_price.plot(kind='box')
train = train[(train.item_price<100000) & (train.item_cnt_day<1001)]
neg_count = train[train.item_price < 0]

neg_count
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median
train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
train.shop_id.unique()
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]

shops.head()

shops.city_code.unique().sum()
cats['split'] = cats['item_category_name'].str.split('-')

cats.head()
cats['type'] = cats['split'].map(lambda x: x[0].strip())

cats.head()

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

cats.head()
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats.head()
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats.head()
cats = cats[['item_category_id','type_code', 'subtype_code']]

cats.head()
items.drop(['item_name'], axis=1, inplace=True)

items.head()
len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test)
matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train[train.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))
train[train.date_block_num ==1 ]
matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix
matrix.sort_values(cols,inplace=True) 

matrix
train['revenue'] = train['item_price'] *  train['item_cnt_day']

train.head()

# calculation of revenue
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group
group.columns = ['item_cnt_month']

group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['item_cnt_month'] = (matrix['item_cnt_month']

                                .fillna(0)  #Replace NaN values by 0

                                .clip(0,20) # NB clip target here

                                .astype(np.float16))

matrix