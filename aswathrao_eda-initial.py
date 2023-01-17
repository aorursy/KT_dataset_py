# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sample = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
train.head().T
train.shape
train.describe()
train.isna().sum()
train.apply(lambda x:len(x.unique()))
train['date'] = pd.to_datetime(train['date'])
train.groupby('item_id')['item_price'].mean()
train[train['item_id']==22167]['item_price'].value_counts()
items.head()
items.duplicated().any() # True
items.groupby('item_category_id')['item_id'].size()
print("Hi")
sns.boxplot(x=train.item_cnt_day)
sns.boxplot(x=train.item_price)
train = train[train.item_price<100000]
train = train[train.item_cnt_day<1001]
train[train.item_price<0]
train = train[train.item_price>=0]
#median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()
#train.loc[train.item_price<0, 'item_price'] = median
#shops are duplicates of each other
shops.shape
shops.duplicated().any() # True
shops.isna().sum()
shops.head()
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])
shops = shops[['shop_id','city_code']]
shops.head()
shop = shops.groupby('city_code')['shop_id'].size().reset_index()
import matplotlib.pyplot as plt
plt.bar(shop['city_code'],shop['shop_id'])
categories.head()
categories[['Item','Category','Extra']] = categories.item_category_name.str.split("-",expand=True) 
categories['Category'].isna().sum()
categories['split'] = categories['item_category_name'].str.split('-')
categories['subtype'] = categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
categories['subtype_code'] = LabelEncoder().fit_transform(categories['subtype'])
#categories = categories[['item_category_id','type_code', 'subtype_code']]
categories = categories [['item_category_id','Item', 'subtype_code']]
categories['category_code'] = LabelEncoder().fit_transform(categories['Item'])
categories = categories[['item_category_id','subtype_code','category_code']]
items.head()
items.drop(['item_name'], axis=1, inplace=True)
#train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
#test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
#sample = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
#items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
##categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
#shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
trainv1= train.join(items, on='item_id', rsuffix='_').join(shops, on='shop_id', rsuffix='_').join(categories,
                                on='item_category_id', rsuffix='_')
trainv2 = trainv1[['date','date_block_num','shop_id','item_id','item_price','item_category_id','city_code','subtype_code','category_code','item_cnt_day']]
trainv2.isna().sum()
test_shop_ids = test['shop_id'].unique()
test_item_ids = test['item_id'].unique()
# Only shops that exist in test set.
lk_train = trainv2[trainv2['shop_id'].isin(test_shop_ids)]
# Only items that exist in test set.
lk_train = lk_train[lk_train['item_id'].isin(test_item_ids)]
lk_train[lk_train.item_cnt_day<0]
