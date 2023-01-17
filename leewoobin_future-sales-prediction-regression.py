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
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from sklearn import ensemble, metrics



# 출력 포맷팅 설정 

# 데이터 값 실수. 소수점 셋째자리까지 표시

pd.options.display.float_format = '{:.3f}'.format
# 날짜/시간 파싱 함수 수동 지정 

# 날짜/시간 파싱하는 포맷을 lambda 함수로 직접 명시적으로 사용자가 지정을 해주어서 

# read_csv() 함수로 파일을 읽어올 때 이 함수를 사용하여 날짜/시간 포맷을 파싱하는 방법



parser = lambda date: pd.to_datetime(date, format='%d.%m.%Y')



train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv', 

                    parse_dates=['date'], date_parser=parser)

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

item_cats = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')



print('train:', train.shape, 'test:', test.shape)

print('items:', items.shape, 'item_cats:', item_cats.shape, 'shops:', shops.shape)
train.head()
test.head()
items.head()
item_cats.head()
shops.head()
# date_block_num - a consecutive month number, used for convenience. 

# January 2013 is 0, February 2013 is 1,..., October 2015 is 33

print(train['date_block_num'].max())
print(train['item_cnt_day'].describe())
train['item_cnt_day'].nlargest(25).values
# isin : 열이 list의 값들을 포함하고 있는 모든 행들을 골라낼 때 사용



test_only = test[~test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()

print('test only items:', len(test_only))

print(test_only)
test_not_only = test[test['item_id'].isin(train['item_id'].unique())]['item_id'].unique()

print('test only items:', len(test_not_only))

print(test_not_only)
# drop duplicates

subset = ['date','date_block_num','shop_id','item_id','item_cnt_day']



print(train.duplicated(subset=subset).value_counts())

train.drop_duplicates(subset=subset, inplace=True)
# drop shops&items not in test data

test_shops = test.shop_id.unique()

test_items = test.item_id.unique()

train = train[train.shop_id.isin(test_shops)] 

train = train[train.item_id.isin(test_items)]



print('train:', train.shape)
from itertools import product



# create all combinations

block_shop_combi = pd.DataFrame(list(product(np.arange(34), test_shops)), 

                                columns=['date_block_num','shop_id'])



shop_item_combi = pd.DataFrame(list(product(test_shops, test_items)), 

                               columns=['shop_id','item_id'])



all_combi = pd.merge(block_shop_combi, shop_item_combi, on=['shop_id'], 

                     how='inner') 

# merge : dataframe 병합 / on : 특정 열을 지정하여 그 열을 기준으로 병합 / 

# inplace=True : 실제로 바뀜 / how='inner' : 중복된 데이터 행만 출력

# how='outer': 기준열 데이터의 중복 여부 상관 없이 모두 출력



print(len(all_combi), 34 * len(test_shops) * len(test_items))





# group by monthly

train_base = pd.merge(all_combi, train, 

                      on=['date_block_num','shop_id','item_id'], how='left')

train_base['item_cnt_day'].fillna(0, inplace=True)

train_grp = train_base.groupby(['date_block_num','shop_id','item_id'])
block_shop_combi
shop_item_combi
all_combi
train_base
# summary count by month

train_monthly = pd.DataFrame(train_grp.agg({'item_cnt_day':

                                            ['sum','count']})).reset_index()



train_monthly.columns = ['date_block_num','shop_id','item_id',

                         'item_cnt','item_order']



print(train_monthly[['item_cnt','item_order']].describe())





# trim count

train_monthly['item_cnt'].clip(0, 20, inplace=True)



train_monthly.head()
# pickup first category name

item_grp = item_cats['item_category_name'].apply(lambda x: str(x).split(' ')[0])

item_cats['item_group'] = pd.Categorical(item_grp).codes

#item_cats = item_cats.join(pd.get_dummies(item_grp, prefix='item_group', drop_first=True))

items = pd.merge(items, item_cats.loc[:,['item_category_id','item_group']], 

                 on=['item_category_id'], how='left')



item_grp.unique()
items
city = shops.shop_name.apply(lambda x: str.

                             replace(x, '!', '')).apply(lambda x: x.split(' ')[0])

shops['city'] = pd.Categorical(city).codes



city.unique()