# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import seaborn as sns

import os

print(os.listdir("../input"))

sns.set(rc={'figure.figsize':(11.7,8.27)})

import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.
shops=pd.read_csv('../input/shops.csv')

train=pd.read_csv('../input/sales_train.csv')

items=pd.read_csv('../input/items.csv')

items_categories=pd.read_csv('../input/item_categories.csv')

test=pd.read_csv('../input/test.csv')

sample=pd.read_csv('../input/sample_submission.csv')

train['date']=pd.to_datetime(train.date,format='%d.%m.%Y')

train.head()
ax = sns.countplot(x="item_cnt_day", data=train[train['item_cnt_day']<0])
train=train.groupby([ 'date_block_num','shop_id','item_id'],sort=True,as_index=False).agg({'item_cnt_day':sum,

                                                             'item_price': 'mean', 

                                                            'date': 'first'}) 
train['month']=train['date'].apply(lambda x : x.month)
np.unique(train.shop_id)
## joint to bring category id 

train=pd.merge(train, items, on='item_id',how='left',suffixes=('_left', '_right'))

train.head()

sells_per_cat=train.groupby(['date_block_num','shop_id','item_category_id'],as_index = False)['item_cnt_day'].sum()

sells_per_cat.columns=['date_block_num','shop_id','item_category_id','item_cnt_category']

sells_per_cat.head()

g = sns.countplot(x="item_category_id", data=train)

all_items_sells=train[['date_block_num','item_cnt_day','item_category_id']]

all_items_sells=all_items_sells.groupby(['date_block_num','item_category_id'],as_index=False).sum()

all_items_sells.head()
for x in np.unique(all_items_sells['item_category_id']):

    sns.lineplot(x='date_block_num',y='item_cnt_day',data=all_items_sells[all_items_sells['item_category_id']==x])

plt.title('Sells count per Category')
all_items_prices=train[['date_block_num','item_price','item_id']]

all_items_prices=all_items_prices.groupby(['date_block_num','item_id'],as_index=False).mean()

all_items_prices.head()
all_items_sells=train[['date_block_num','item_cnt_day','item_category_id']]

all_items_sells=all_items_sells.groupby(['date_block_num','item_category_id'],as_index=False).sum()

all_items_sells.head()
validation = train[train['date_block_num']==33]

train = train[train['date_block_num']<33]
True_labels=validation['item_price']
validation=validation.reset_index(drop=True)
category_median= train[['item_category_id','item_price']].groupby(['item_category_id']).median()

category_median.head()
last_item_price_same_shop= train[['shop_id','item_id','item_price']].groupby(['shop_id','item_id'],as_index=False).last()

last_item_price_same_shop.head()

#last_item_price_same_shop[(last_item_price_same_shop['shop_id']==0 )& (last_item_price_same_shop['item_id']==33 )]
last_item_price_all_shop= train[['item_id','item_price']].groupby(['item_id'],as_index=False).last()

last_item_price_all_shop.head()

## redo this 

%time

def get_last_price_same_shop(row):

    

    #return np.mean(category_median['item_price'])

    return 1

validation['item_price']=validation.apply(lambda x : get_last_price_same_shop(x),axis=1)

validation.head()