# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from sklearn import *

import nltk, datetime

import seaborn as sns



import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')

submission = pd.read_csv('../input/sample_submission.csv')

items = pd.read_csv('../input/items.csv')

item_categories = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')

print('train:', train.shape, 'test:', test.shape)
train.head()
submission.head()
train.item_cnt_day.plot()

plt.title("Number of products sold per day");
train.item_price.hist()

plt.title("Item Price Distribution");
from wordcloud import WordCloud

import random



def grey_color_func(word, font_size, position, orientation, random_state=None, **kwargs):

    return "hsl(0, 0%%, %d%%)" % random.randint(60, 100)



item = ' '.join(items.item_name).lower()

# wordcloud for display address

plt.figure(figsize=(12,6))

wc = WordCloud(background_color='gold', max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=400,

                            relative_scaling=.5).generate(item)

plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3))

#plt.imshow(wc)

plt.title("Items", fontsize=20)

plt.savefig('items-wordcloud.png')

plt.axis("off");
from wordcloud import WordCloud

import random





item_cat = ' '.join(item_categories.item_category_name).lower()

# wordcloud for display address

plt.figure(figsize=(12,6))

wc = WordCloud(background_color='black', max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=400,

                            relative_scaling=.5).generate(item_cat)

plt.imshow(wc)

#plt.imshow(wc)

plt.title("Items Categories", fontsize=20)

plt.savefig('items-cat-wordcloud.png')

plt.axis("off");
from wordcloud import WordCloud

import random





shop = ' '.join(shops.shop_name).lower()

# wordcloud for display address

plt.figure(figsize=(12,6))

wc = WordCloud(background_color='white', max_font_size=200,

                            width=1600,

                            height=800,

                            max_words=400,

                            relative_scaling=.5).generate(shop)

plt.imshow(wc)

#plt.imshow(wc)

plt.title("shops", fontsize=20)

plt.savefig('shops-wordcloud.png')

plt.axis("off");
#Make Monthly

train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')

train['month'] = train['date'].dt.month

train['year'] = train['date'].dt.year

train = train.drop(['date','item_price'], axis=1)

train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()

train = train.rename(columns={'item_cnt_day':'item_cnt_month'})

#Monthly Mean

shop_item_monthly_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean()

shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})

#Add Mean Feature

train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])

#Last Month (Oct 2015)

shop_item_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]

shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})

shop_item_prev_month.head()

#Add Previous Month Feature

train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)

#Items features

train = pd.merge(train, items, how='left', on='item_id')

#Item Category features

train = pd.merge(train, item_categories, how='left', on='item_category_id')

#Shops features

train = pd.merge(train, shops, how='left', on='shop_id')

test['month'] = 11

test['year'] = 2015

test['date_block_num'] = 34

#Add Mean Feature

test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id','item_id']).fillna(0.)

#Add Previous Month Feature

test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)

#Items features

test = pd.merge(test, items, how='left', on='item_id')

#Item Category features



test = pd.merge(test, item_categories, how='left', on='item_category_id')

#Shops features

test = pd.merge(test, shops, how='left', on='shop_id')

test['item_cnt_month'] = 0.
