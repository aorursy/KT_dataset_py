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
import matplotlib.pyplot as plt

import seaborn as sns
shops         = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

test          = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

sales_train   = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

item_cat      = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

items         = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
def isNullHeatMap(df, figsize = (15, 7)):

    print('null count is {}'.format(df.isnull().sum()))

    plt.figure(figsize = figsize)

    sns.heatmap(df.isnull())
def getUniqueCount(df):

    columns = df.columns

    for col in columns:

        print('unique value of {} is {}'.format(col, len(df[col].unique())))
shops.head()
isNullHeatMap(shops)
test.head()
isNullHeatMap(test)
sales_train.head()
isNullHeatMap(sales_train)
item_cat.head()
isNullHeatMap(item_cat)
items.head()
isNullHeatMap(items)
shops.info()
getUniqueCount(shops)
col = 'shop_name'

for index in range(shops.shape[0]):

    print(str(index) + ' ' + shops.loc[index, col])
shops['shop_name'] = shops['shop_name'].apply(lambda x : x[1:] if x[0] == '!' else x)

shops['city'] = shops['shop_name'].apply(lambda x : x.split(' ')[0])
from sklearn.preprocessing import LabelEncoder
shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops.drop(['shop_name', 'city'], axis = 1, inplace = True)
shops.head()
item_cat.info()
getUniqueCount(item_cat)
col = 'item_category_name'

for index in range(item_cat.shape[0]):

    print(str(index) + ' ' + item_cat.loc[index, col])
item_cat['item_cat'] = item_cat['item_category_name'].apply(lambda x : x.split(' - '))
item_cat['item_cat'].isnull().sum()
item_cat['item_sub_category'] = item_cat['item_cat'].apply(lambda x : x[1] if len(x) > 1 else 'N/A')

item_cat['item'] = item_cat['item_cat'].apply(lambda x : x[0] if len(x) > 0 else 'N/A')
item_cat.loc[item_cat['item_category_name'] == 'Билеты (Цифра)','item_sub_category'] = 'Digital'

item_cat.loc[item_cat['item_category_name'] == 'Доставка товара','item_sub_category'] = 'Unknown'

item_cat.loc[item_cat['item_category_name'] == 'Карты оплаты (Кино, Музыка, Игры)','item_sub_category'] = 'Cinema, Music, Games'

item_cat.loc[item_cat['item_category_name'] == 'Служебные','item_sub_category'] = 'Unknown'



item_cat.loc[item_cat['item_category_name'] == 'Чистые носители (шпиль)','item_sub_category'] = 'spire'

item_cat.loc[item_cat['item_category_name'] == 'Чистые носители (штучные)','item_sub_category'] = 'piece'

item_cat.loc[item_cat['item_category_name'] == 'Элементы питания','item_sub_category'] = 'Unknown'

item_cat[item_cat['item_sub_category'] == 'N/A']['item_category_name']
item_cat[item_cat['item_sub_category'] == 'N/A']['item_category_name']
item_cat.head()
#drop other columns

item_cat.drop(['item_category_name', 'item_cat'], axis = 1, inplace = True)
item_cat['item'] = LabelEncoder().fit_transform(item_cat['item'])

item_cat['item_sub_category'] = LabelEncoder().fit_transform(item_cat['item_sub_category'])
item_cat.head()
items.head()
getUniqueCount(items)
items.drop(['item_name'], axis = 1, inplace = True)
items.head()
sales_train.head()
plt.figure(figsize=(15, 7))

sns.lineplot(data=sales_train, x = 'date_block_num', y = 'item_cnt_day')
index_col = ['date_block_num', 'shop_id', 'item_id']
sales_train.head()
## aggregate train data with date_block

total_train_df = sales_train.groupby(index_col).agg({'item_price' : ['mean'], 'item_cnt_day' : ['sum']})

total_train_df.reset_index(inplace = True)

total_train_df.columns = ['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_month']

## test data to train format

total_test_df = test.copy()

total_test_df['ID'] = 34

total_test_df.columns = index_col
## concate train and test dataframe

all_df = pd.concat([total_train_df,total_test_df], ignore_index = True, sort = False)
## show all data

all_df.head()
all_data = all_df.copy()
## aggregate item prrice data

aggregate = ['mean', 'max', 'min', 'std']

columns = ['item_price', 'item_cnt_month']

index_col = ['date_block_num', 'item_id']

for col in columns:

    for agg in aggregate:

        agg_column = col + '_' + agg

        df = all_data.groupby(['date_block_num', 'item_id']).agg({col : [agg]})

        df.columns = [agg_column]

        df.reset_index(inplace = True)

        all_data = all_data.merge(df, on = ['date_block_num', 'item_id'], how = 'left')
all_data.head()
lagging = [1, 3, 4, 6, 12]

index_col = ['date_block_num', 'shop_id', 'item_id']

lag_columns = ['item_price', 'item_cnt_month',

       'item_price_mean', 'item_price_max', 'item_price_min', 'item_price_std',

       'item_cnt_month_mean', 'item_cnt_month_max', 'item_cnt_month_min',

       'item_cnt_month_std']

for col in lag_columns:

    for lag in lagging:

        df = all_data[['date_block_num', 'shop_id', 'item_id', col]].copy()

        lag_column = col + '_' + str(lag)

        df.columns = ['date_block_num', 'shop_id', 'item_id', lag_column]

        df['date_block_num'] = df['date_block_num'] + lag

        all_data = all_data.merge(df, on = index_col, how = 'left')
all_data.head()
## include data date_block_num greater than 12

all_data = all_data[all_data['date_block_num'] >= 12].reset_index(drop = True)
plt.figure(figsize = (15, 7))

sns.heatmap(all_data.isnull())
## merge all dataset

all_data = all_data.merge(shops, on = ['shop_id'], how = 'left')

all_data = all_data.merge(items, on = ['item_id'], how = 'left')

all_data = all_data.merge(item_cat, on = ['item_category_id'], how = 'left')
## there are few shops are duplicated

all_data.loc[all_data['shop_id'] ==  57, 'shop_id'] = 0

all_data.loc[all_data['shop_id'] ==  58, 'shop_id'] = 1

all_data.loc[all_data['shop_id'] ==  11, 'shop_id'] = 10

all_data.loc[all_data['shop_id'] ==  24, 'shop_id'] = 23

all_data.loc[all_data['shop_id'] ==  40, 'shop_id'] = 39

all_data.loc[all_data['shop_id'] ==  41, 'shop_id'] = 39
all_data.head()
## fill na with 0

all_data.fillna(0, inplace = True)
all_data.head()
train_all_df = all_data[all_data['date_block_num'] <= 33].drop(['item_price'], axis = 1)

test_all_df  = all_data[all_data['date_block_num']  == 34].drop(['item_price', 'item_cnt_month'], axis = 1).reset_index(drop = True)
train_all_df.head()
X_train_df = train_all_df[train_all_df['date_block_num'] < 33].drop(['item_cnt_month'], axis = 1)

Y_train_df = train_all_df[train_all_df['date_block_num'] < 33]['item_cnt_month']



X_val_df = train_all_df[train_all_df['date_block_num'] == 33].drop(['item_cnt_month'], axis = 1)

Y_val_df = train_all_df[train_all_df['date_block_num'] == 33]['item_cnt_month']