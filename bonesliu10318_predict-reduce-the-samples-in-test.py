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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from scipy import stats

from scipy.stats import norm

import datetime

import warnings

import os

warnings.filterwarnings('ignore')



%matplotlib inline
path = '../input/competitive-data-science-predict-future-sales/'



os.listdir(path)
items = pd.read_csv(path + 'items.csv')

item_categories = pd.read_csv(path + 'item_categories.csv')

train = pd.read_csv(path + 'sales_train.csv')

shop = pd.read_csv(path + 'shops.csv')

test = pd.read_csv(path + 'test.csv')
print('The information of the dataset'.center(50, '-'))

print('The shape of the dataset is {}'.format(items.shape))

print('The number of the goods {}'.format(items.item_id.nunique()))

print('The category of the goods {}'.format(items.item_category_id.nunique()))

items.sample(4)
# The missing value of the dataset

total = items.isnull().sum()

percentage = total / items.shape[0]

types = items.dtypes

pd.concat([total, percentage, types], axis = 1, keys = ['Total', 'Percentage', 'Types'])
print('The information of the dataset'.center(50, '-'))

print('The shape of the dataset is {}'.format(train.shape))

print('The number of the goods {}'.format(train.item_id.nunique()))

train.sample(4)

# the numner of the goods 21807 mean that some goods hasn't sale at all.
train['date'] = train['date'].apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))
# The missing value of the dataset

total = train.isnull().sum()

percentage = total / train.shape[0]

types = train.dtypes

pd.concat([total, percentage, types], axis = 1, keys = ['Total', 'Percentage', 'Types'])
print('The information of the shop dataset'.center(50, '-'))

print('The shape of the shop dataset {}'.format(shop.shape))

print('The shop information {} and the colums {}'.format(shop.nunique(), shop.columns))
print('The information of the item_categories dataset'.center(80, '-'))

print('The shape of the shop dataset {}'.format(item_categories.shape))

print('The shop information {} and the colums {}'.format(item_categories.nunique(), item_categories.columns))
salesData = pd.merge(train, items, how = 'inner', on = 'item_id')

salesData = pd.merge(salesData, shop, how = 'inner', on = 'shop_id')

salesData = pd.merge(salesData, item_categories, how = 'inner', on = 'item_category_id')
temp_shop_name = salesData['shop_name']

salesData.drop('shop_name', axis = 1, inplace = True)

salesData.insert(3, 'shop_name', temp_shop_name)
test.sample(4)
# It mean that the test data samples less than train data

print(train['shop_id'].nunique() is test['shop_id'].nunique())

temp_shop_id = test['shop_id'].unique()

temp_item_id = test['item_id'].unique()

train_lk = salesData[salesData['shop_id'].isin(temp_shop_id)]

train_lk = train_lk[train_lk['item_id'].isin(temp_item_id)]

print('Before'.center(50, '-'))

print('The shape of the train {}'.format(train.shape))

print('After'.center(50, '-'))

print('The shape ot the train_lk {}, and the leakage samples are {}'.format(train_lk.shape, train.shape[0] - train_lk.shape[0]))
salesData.isnull().sum()
print(items['item_id'].nunique() - train['item_id'].nunique())
train_item_id = train['item_id'].unique()

# ~ mean that turn the false to true

items_lk = items[~items['item_id'].isin(train_item_id)]

temp_items_lk = items_lk['item_id']

test_lk = test[test['item_id'].isin(temp_items_lk)]

print('The shape of the test_lk is {}, and the predicted value of this sample can be set to 0'.format(test_lk.shape[0]))
test.shape
salesData.sample(4)
train_monthly = train_lk[['date', 'date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'item_cnt_day']]
train_monthly = train_monthly.loc[train_monthly['item_price'] > 0]
train_monthly = train_monthly.sort_values(by = 'date').groupby(['date_block_num', 'shop_id', 'item_category_id', 'item_id']).agg({'item_price': ['sum', 'mean'], 'item_cnt_day': ['sum', 'mean', 'count']}).reset_index()
train_monthly.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']
train_monthly.sample(4)
def handleMonth(date):

    if date < 12:

        month = date + 1

    elif date % 12 == 0:

        month = 12

    else:

        month = (date % 12) + 1

    return month
train_monthly['year'] = train_monthly['date_block_num'].apply(lambda x: ((x//12) + 2013))

train_monthly['month'] = train_monthly['date_block_num'].apply(handleMonth)
train_monthly.sample(4)
month_mean = train_monthly.groupby(['month'])['item_cnt'].mean().reset_index()

month_sum = train_monthly.groupby(['month'])['item_cnt'].sum().reset_index()



month_data = pd.merge(month_mean, month_sum, how = 'inner', on = 'month')



category_mean = train_monthly.groupby(['item_category_id'])['item_cnt'].mean().reset_index()

category_sum = train_monthly.groupby(['item_category_id'])['item_cnt'].sum().reset_index()



category_data = pd.merge(category_mean, category_sum, how = 'inner', on = 'item_category_id')



shop_mean = train_monthly.groupby(['shop_id'])['item_cnt'].mean().reset_index()

shop_sum = train_monthly.groupby(['shop_id'])['item_cnt'].sum().reset_index()



shop_data = pd.merge(shop_mean, shop_sum, how = 'inner', on = 'shop_id')
month_data.sample(4)
fig, axes = plt.subplots(1, 2, figsize = (22, 4))

sns.set_style('whitegrid')

ax1 = sns.pointplot(x = 'month', y = 'item_cnt_x', data = month_data, linestyles = '-', ax = axes[0])

ax2 = sns.pointplot(x = 'month', y = 'item_cnt_y', data = month_data, linestyles = '-', ax = axes[1])

ax1.set(title = 'mothly mean', ylabel = 'item_cnt')

ax2.set(title = 'mothly sum', ylabel = 'item_cnt')
category_data.sample(4)
fig, axes = plt.subplots(2, 1, figsize = (18, 8))

sns.set_style('whitegrid')

ax1 = sns.barplot(x = 'item_category_id', y = 'item_cnt_x', data = category_data, ax = axes[0])

ax2 = sns.barplot(x = 'item_category_id', y = 'item_cnt_y', data = category_data, ax = axes[1])

ax1.set(title = 'category mean', ylabel = 'item_cnt')

ax2.set(title = 'category sum', ylabel = 'item_cnt')
shop_data.sample(4)
fig, axes = plt.subplots(2, 1, figsize = (18, 8), sharex = True)

sns.set_style('whitegrid')

ax1 = sns.barplot(x = 'shop_id', y = 'item_cnt_x', data = shop_data, ax = axes[0], palette = 'mako')

ax2 = sns.barplot(x = 'shop_id', y = 'item_cnt_y', data = shop_data, ax = axes[1], palette = 'mako')

ax1.set(title = 'shop mean', ylabel = 'item_cnt')

ax2.set(title = 'shop sum', ylabel = 'item_cnt')
train_monthly = train_monthly.loc[train_monthly['item_cnt'] >= 0].loc[train_monthly['item_cnt'] <= 20].loc[train_monthly['item_price'] < 400000]

# tht following code much better than the above

# train_monthly = train_monthly.query('item_cnt >= 0 and item_cnt <= 20 and item_price < 400000')
train_monthly['item_price_unit'] = train_monthly['item_price'] // train_monthly['item_cnt']
gp_item_price = train_monthly.sort_values('date_block_num').groupby(['item_id']).agg({'item_price':[np.min, np.max]}).reset_index()

gp_item_price.columns = ['item_id', 'hist_min_item_price', 'hist_max_item_price']



train_monthly = pd.merge(train_monthly, gp_item_price, on='item_id', how='inner')
train_monthly['price_increase'] = train_monthly['item_price'] - train_monthly['hist_min_item_price']

train_monthly['price_decrease'] = train_monthly['hist_max_item_price'] - train_monthly['item_price']
# Min value

f_min = lambda x: x.rolling(window=3, min_periods=1).min()

# Max value

f_max = lambda x: x.rolling(window=3, min_periods=1).max()

# Mean value

f_mean = lambda x: x.rolling(window=3, min_periods=1).mean()

# Standard deviation

f_std = lambda x: x.rolling(window=3, min_periods=1).std()



function_list = [f_min, f_max, f_mean, f_std]

function_name = ['min', 'max', 'mean', 'std']



for i in range(len(function_list)):

    train_monthly[('item_cnt_%s' % function_name[i])] = train_monthly.sort_values('date_block_num').groupby(['shop_id', 'item_category_id', 'item_id'])['item_cnt'].apply(function_list[i])



# Fill the empty std features with 0

train_monthly['item_cnt_std'].fillna(0, inplace=True)
print('The shape of the dataset {}'.format(train_monthly.shape))

train_monthly.sample(4)
# maybe some added some codes later