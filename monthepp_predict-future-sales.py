# import python standard library

import gc, itertools



# import data manipulation library

import numpy as np

import pandas as pd



# import data visualization library

import matplotlib.pyplot as plt

import seaborn as sns



# import sklearn data preprocessing

from sklearn.preprocessing import LabelEncoder



# import xgboost model class

import xgboost as xgb



# import sklearn model selection

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import train_test_split



# import sklearn model evaluation regression metrics

from sklearn.metrics import mean_squared_error
# pandas options

pd.options.display.max_rows = 10
# acquiring training and testing data

df_train = pd.read_csv('../input/sales_train.csv')

df_test = pd.read_csv('../input/test.csv')
# acquiring supplemental information

df_items = pd.read_csv('../input/items.csv')

df_categories = pd.read_csv('../input/item_categories.csv')

df_shops = pd.read_csv('../input/shops.csv')
# visualize head of the training data

df_train.head(n=5)
# visualize tail of the testing data

df_test.tail(n=5)
# visualize head of the supplemental information about the items/products

df_items.head(n=5)
# visualize head of the supplemental information about the items categories

df_categories.head(n=5)
# visualize head of the supplemental information about the shops

df_shops.head(n=5)
# combine training and testing dataframe

df_train['datatype'], df_test['datatype'] = 'training', 'testing'

df_train.insert(0, 'ID', np.nan)

df_test.insert(1, 'date', '01.11.2015')

df_test.insert(2, 'date_block_num', 34)

df_test.insert(df_test.shape[1] - 1, 'item_price', np.nan)

df_test.insert(df_test.shape[1] - 1, 'item_cnt_day', np.nan)

df_data = pd.concat([df_train, df_test], ignore_index=False)
# describe training and testing data

df_data.describe(include='all')
# list all features type number

col_number = df_data.select_dtypes(include=['number']).columns.tolist()

print('features type number:\n items %s\n length %d' %(col_number, len(col_number)))



# list all features type object

col_object = df_data.select_dtypes(include=['object']).columns.tolist()

print('features type object:\n items %s\n length %d' %(col_object, len(col_object)))
# feature exploration: histogram of all numeric features

_ = df_data.hist(bins=20, figsize=(10, 6))
# feature extraction: fix the duplicated shop id

df_data.loc[df_data['shop_id'] == 0, 'shop_id'] = 57

df_data.loc[df_data['shop_id'] == 1, 'shop_id'] = 58

df_data.loc[df_data['shop_id'] == 11, 'shop_id'] = 10
# feature extraction: set maximum and minimum limit for item price

df_data.loc[df_data['item_price'] < 0, 'item_price'] = df_data.loc[(df_data['date_block_num'] == 4) & (df_data['shop_id'] == 32) & (df_data['item_price'] > 0), 'item_price'].median()

df_data = df_data[((df_data['item_price'] >= 0) & (df_data['item_price'] <= 100000)) | (df_data['item_price'].isna())]
# feature extraction: set maximum and minimum limit for number of products sold

df_data = df_data[(df_data['item_cnt_day'] <= 1000) | (df_data['item_cnt_day'].isna())]
# feature exploration: histogram of all numeric features

_ = df_data.hist(bins=20, figsize=(10, 6))
# feature exploration: zero number of products sold

df_data[df_data['item_cnt_day'] == 0].head()
# feature extraction: cross dataframe

list_of_cross = []

for dateblocknum in df_data['date_block_num'].unique():

    shops = df_data.loc[df_data['date_block_num'] == dateblocknum, 'shop_id'].unique()

    items = df_data.loc[df_data['date_block_num'] == dateblocknum, 'item_id'].unique()

    list_of_cross.append(np.array(list(itertools.product(*[[dateblocknum], shops, items]))))

df_cross = pd.DataFrame(np.vstack(list_of_cross), columns=['date_block_num', 'shop_id', 'item_id'])
# describe cross dataframe

df_cross.describe(include='all')
# feature extraction: block dataframe

df_block = df_data.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False).agg({

    'item_price': 'mean', 'item_cnt_day': sum

}).rename(columns={'item_cnt_day': 'item_cnt_month'})
# describe block dataframe for month 0 - 33

df_block[df_block['date_block_num'] != 34].describe(include='all')
# describe block dataframe for month 34

df_block[df_block['date_block_num'] == 34].describe(include='all')
# feature extraction: merge block and cross dataframe

df_block = pd.merge(df_cross, df_block, how='left', left_on=['date_block_num', 'shop_id', 'item_id'], right_on=['date_block_num', 'shop_id', 'item_id'])
# feature extraction: number of products sold fillna by 0

df_block['item_cnt_month'] = df_block['item_cnt_month'].fillna(0)
# feature extraction: merge with supplemental information about the items/products

df_block = pd.merge(df_block, df_items, how='left', left_on='item_id', right_on='item_id')
# feature extraction: merge with supplemental information about the items categories

df_block = pd.merge(df_block, df_categories, how='left', left_on='item_category_id', right_on='item_category_id')
# feature extraction: merge with supplemental information about the shops

df_block = pd.merge(df_block, df_shops, how='left', left_on='shop_id', right_on='shop_id')
# feature exploration: item id

df_block['item_id'].value_counts()
# feature exploration: category id

df_block['item_category_id'].value_counts()
# feature exploration: shop id

df_block['shop_id'].value_counts()
# feature exploration: item id 20949

df_items.loc[df_items['item_id'] == 20949, 'item_category_id']
# feature exploration: category id 71

df_items.loc[df_items['item_category_id'] == 71, 'item_id']
# feature exploration: shop id 31

df_block.loc[df_block['shop_id'] == 31, 'item_category_id'].value_counts()
# feature exploration: category id 40

df_block.loc[df_block['item_category_id'] == 40, 'item_id'].value_counts()
# feature exploration: category id 40

df_block.loc[df_block['item_category_id'] == 40, 'shop_id'].value_counts()
# feature exploration: item id happened in month 33

df_block.loc[df_block['date_block_num'] == 33, 'item_id'][~df_block.loc[df_block['date_block_num'] == 33, 'item_id'].isin(df_block.loc[df_block['date_block_num'] < 33, 'item_id'])].value_counts()
# feature exploration: item id happened in month 34

df_block.loc[df_block['date_block_num'] == 34, 'item_id'][~df_block.loc[df_block['date_block_num'] == 34, 'item_id'].isin(df_block.loc[df_block['date_block_num'] < 34, 'item_id'])].value_counts()
# feature exploration: category id happened in month 33

df_block.loc[df_block['date_block_num'] == 33, 'item_category_id'][~df_block.loc[df_block['date_block_num'] == 33, 'item_category_id'].isin(df_block.loc[df_block['date_block_num'] < 33, 'item_category_id'])].value_counts()
# feature exploration: category id happened in month 34

df_block.loc[df_block['date_block_num'] == 34, 'item_category_id'][~df_block.loc[df_block['date_block_num'] == 34, 'item_category_id'].isin(df_block.loc[df_block['date_block_num'] < 34, 'item_category_id'])].value_counts()
# feature exploration: shop id happened in month 33

df_block.loc[df_block['date_block_num'] == 33, 'shop_id'][~df_block.loc[df_block['date_block_num'] == 33, 'shop_id'].isin(df_block.loc[df_block['date_block_num'] < 33, 'shop_id'])].value_counts()
# feature exploration: shop id happened in month 34

df_block.loc[df_block['date_block_num'] == 34, 'shop_id'][~df_block.loc[df_block['date_block_num'] == 34, 'shop_id'].isin(df_block.loc[df_block['date_block_num'] < 34, 'shop_id'])].value_counts()
# memory clean-up

del df_categories, df_cross, df_data, df_items, df_shops

gc.collect()
# feature extraction: year

df_block['year'] = df_block['date_block_num'] // 12
# feature extraction: month

df_block['month'] = df_block['date_block_num'] % 12
# feature extraction: day

day = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])

df_block['day'] = df_block['month'].map(day)
# feature extraction: city

df_block.loc[df_block['shop_name'] == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

df_block['city'] = df_block['shop_name'].str.split(' ').apply(lambda x: x[0])

df_block.loc[df_block['city'] == '!Якутск', 'city'] = 'Якутск'

df_block['city'].value_counts()
# feature extraction: city_code

df_block['city_code'] = LabelEncoder().fit_transform(df_block['city'])
# feature extraction: type

df_block['type'] = df_block['item_category_name'].str.split('-').apply(lambda x: x[0].strip())

df_block['type'].value_counts()
# feature extraction: type_code

df_block['type_code'] = LabelEncoder().fit_transform(df_block['type'])
# feature extraction: subtype

df_block['subtype'] = df_block['item_category_name'].str.split('-').apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

df_block['subtype'].value_counts()
# feature extraction: subtype_code

df_block['subtype_code'] = LabelEncoder().fit_transform(df_block['subtype'])
# describe block dataframe for month 0 - 33

df_block[df_block['date_block_num'] != 34].describe(include='all')
# describe block dataframe for month 34

df_block[df_block['date_block_num'] == 34].describe(include='all')
# feature exploration: item price and number of products sold by item id 20949

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_id'] == 20949].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_id'] == 20949], ax=axes[1])
# feature exploration: item price and number of products sold by category id 71

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_category_id'] == 71].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_category_id'] == 71], ax=axes[1])
# feature exploration: item price and number of products sold by item id 8778

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_id'] == 8778].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_id'] == 8778], ax=axes[1])
# feature exploration: item price and number of products sold by item id 8778 and shop id 31

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[(df_block['item_id'] == 8778) & (df_block['shop_id'] == 31)].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[(df_block['item_id'] == 8778) & (df_block['shop_id'] == 31)], ax=axes[1])
# feature exploration: item price and number of products sold by item id 8778 and shop id 25

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[(df_block['item_id'] == 8778) & (df_block['shop_id'] == 25)].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[(df_block['item_id'] == 8778) & (df_block['shop_id'] == 25)], ax=axes[1])
# feature exploration: item price and number of products sold by item id 19602

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_id'] == 19602].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_id'] == 19602], ax=axes[1])
# feature exploration: item price and number of products sold by item id 19602 and shop id 31

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[(df_block['item_id'] == 19602) & (df_block['shop_id'] == 31)].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[(df_block['item_id'] == 19602) & (df_block['shop_id'] == 31)], ax=axes[1])
# feature exploration: item price and number of products sold by item id 19602 and shop id 25

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[(df_block['item_id'] == 19602) & (df_block['shop_id'] == 25)].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[(df_block['item_id'] == 19602) & (df_block['shop_id'] == 25)], ax=axes[1])
# feature exploration: item price and number of products sold by category id 40

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[df_block['item_category_id'] == 40].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[df_block['item_category_id'] == 40], ax=axes[1])
# feature exploration: item price and number of products sold by category id 40 and shop id 31

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[(df_block['item_category_id'] == 40) & (df_block['shop_id'] == 31)].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[(df_block['item_category_id'] == 40) & (df_block['shop_id'] == 31)], ax=axes[1])
# feature exploration: item price and number of products sold by category id 40 and shop id 25

fig, axes = plt.subplots(figsize=(20 , 6), ncols=1, nrows=2)

axes = axes.flatten()

sns.scatterplot(x='date_block_num', y='item_cnt_month', data=df_block[(df_block['item_category_id'] == 40) & (df_block['shop_id'] == 25)].groupby(['date_block_num'], as_index=False).agg({'item_cnt_month': 'mean'}), ax=axes[0])

sns.boxplot(x='date_block_num', y='item_price', data=df_block[(df_block['item_category_id'] == 40) & (df_block['shop_id'] == 25)], ax=axes[1])
# memory clean-up

df_block = df_block.drop(['item_name', 'item_category_name', 'shop_name', 'city', 'type', 'subtype'], axis=1)

gc.collect()
# feature extraction: shifted features for item price

for i in [1, 2, 3, 4, 5, 6]:

    shifted = df_block[['date_block_num', 'shop_id', 'item_id', 'item_price']].copy(deep=True)

    shifted.columns = ['date_block_num', 'shop_id','item_id', 'item_price_shift' + str(i)]

    shifted['date_block_num'] = shifted['date_block_num'] + i

    df_block = pd.merge(df_block, shifted, how='left', on=['date_block_num', 'shop_id', 'item_id'])
# feature extraction: statistic features for item price by item

df_block['item_price_by_item_mean'] = df_block.groupby(['item_id'])['item_price'].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for item price by item and date block

for i in [1, 2, 3, 4, 5, 6]: df_block['item_price_by_item_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num', 'item_id'])['item_price_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for delta item price by item and date block

for i in [1, 2, 3, 4, 5, 6]: df_block['delta_item_price_by_item_date_mean_shift' + str(i)] = (df_block['item_price_by_item_date_mean_shift' + str(i)] - df_block['item_price_by_item_mean']) / df_block['item_price_by_item_mean']



def select_nonnull(row):

    for i in [1, 2, 3, 4, 5, 6]:

        if not(np.isnan(row['delta_item_price_by_item_date_mean_shift' + str(i)])): return row['delta_item_price_by_item_date_mean_shift' + str(i)]

    return 0

df_block['delta_item_price_by_item_date_mean_shift'] = df_block.apply(select_nonnull, axis=1)
# memory clean-up

df_block = df_block.drop(['item_price_by_item_date_mean_shift' + str(i) for i in [1, 2, 3, 4, 5, 6]], axis=1)

df_block = df_block.drop(['delta_item_price_by_item_date_mean_shift' + str(i) for i in [1, 2, 3, 4, 5, 6]], axis=1)

gc.collect()
# feature extraction: set maximum and minimum limit for number of products sold

df_block['item_cnt_month'] = df_block['item_cnt_month'].clip(0 ,20)
# feature extraction: shifted features for number of products sold

for i in [1, 2, 3, 6, 12]:

    shifted = df_block[['date_block_num', 'shop_id', 'item_id', 'item_cnt_month']].copy(deep=True)

    shifted.columns = ['date_block_num', 'shop_id','item_id', 'item_cnt_month_shift' + str(i)]

    shifted['date_block_num'] = shifted['date_block_num'] + i

    df_block = pd.merge(df_block, shifted, how='left', on=['date_block_num', 'shop_id', 'item_id'])
# feature extraction: statistic shifted features for number of products sold by date block

for i in [1]: df_block['item_cnt_month_by_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num'])['item_cnt_month_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for number of products sold by item and date block

for i in [1, 2, 3, 6, 12]: df_block['item_cnt_month_by_item_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num', 'item_id'])['item_cnt_month_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for number of products sold by category and date block

for i in [1, 2, 3, 6, 12]: df_block['item_cnt_month_by_category_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num', 'item_category_id'])['item_cnt_month_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for number of products sold by shop and date block

for i in [1, 2, 3, 6, 12]: df_block['item_cnt_month_by_shop_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num', 'shop_id'])['item_cnt_month_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for number of products sold by city and date block

for i in [1]: df_block['item_cnt_month_by_city_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num', 'city_code'])['item_cnt_month_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for number of products sold by item, city and date block

for i in [1]: df_block['item_cnt_month_by_item_city_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num', 'item_id', 'city_code'])['item_cnt_month_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: statistic shifted features for number of products sold by category, shop and date block

for i in [1]: df_block['item_cnt_month_by_category_shop_date_mean_shift' + str(i)] = df_block.groupby(['date_block_num', 'item_category_id', 'shop_id'])['item_cnt_month_shift' + str(i)].transform('mean').astype(np.float16)
# feature extraction: first sale

df_block['first_sale_item'] = (df_block['date_block_num'] - df_block.groupby(['item_id'])['date_block_num'].transform('min')).astype(np.int16)

df_block['first_sale_item_shop'] = (df_block['date_block_num'] - df_block.groupby(['item_id', 'shop_id'])['date_block_num'].transform('min')).astype(np.int16)
# feature extraction: number of products sold for first sale by category

list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id'])
# feature extraction: number of products sold for first sale by category for month onwards

for i in [12, 18, 24, 30]:

    list_of_first = []

    for dateblocknum in df_block['date_block_num'].unique():

        df_first = df_block[(df_block['date_block_num'] >= i) & (df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id'], as_index=False).agg({'item_cnt_month': 'mean'})

        df_first.insert(0, 'date_block_num', dateblocknum)

        list_of_first.append(df_first)

    df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_first_month' + str(i) + 'onwards'})

    df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id'])
# feature extraction: number of products sold for first sale by category and shop

list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id', 'shop_id'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_shop_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id', 'shop_id'])
# feature extraction: number of products sold for first sale by category and shop for month onwards

for i in [12, 18, 24, 30]:

    list_of_first = []

    for dateblocknum in df_block['date_block_num'].unique():

        df_first = df_block[(df_block['date_block_num'] >= i) & (df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id', 'shop_id'], as_index=False).agg({'item_cnt_month': 'mean'})

        df_first.insert(0, 'date_block_num', dateblocknum)

        list_of_first.append(df_first)

    df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_shop_first_month' + str(i) + 'onwards'})

    df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id', 'shop_id'])
# feature extraction: number of products sold for first sale by category and city

list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id', 'city_code'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_city_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id', 'city_code'])
# feature extraction: number of products sold for first sale by category and city for month onwards

for i in [12, 18, 24, 30]:

    list_of_first = []

    for dateblocknum in df_block['date_block_num'].unique():

        df_first = df_block[(df_block['date_block_num'] >= i) & (df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id', 'city_code'], as_index=False).agg({'item_cnt_month': 'mean'})

        df_first.insert(0, 'date_block_num', dateblocknum)

        list_of_first.append(df_first)

    df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_city_first_month' + str(i) + 'onwards'})

    df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id', 'city_code'])
# feature extraction: number of products sold for first sale by category and type

list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id', 'type_code'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_type_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id', 'type_code'])
# feature extraction: number of products sold for first sale by category and subtype

list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['item_category_id', 'subtype_code'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_category_subtype_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'item_category_id', 'subtype_code'])
# feature extraction: number of products sold for first sale by shop

list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['shop_id'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_shop_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'shop_id'])
# feature extraction: number of products sold for first sale by city

list_of_first = []

for dateblocknum in df_block['date_block_num'].unique():

    df_first = df_block[(df_block['date_block_num'] < dateblocknum) & (df_block['first_sale_item'] == 0)].groupby(['city_code'], as_index=False).agg({'item_cnt_month': 'mean'})

    df_first.insert(0, 'date_block_num', dateblocknum)

    list_of_first.append(df_first)

df_first = pd.concat(list_of_first, ignore_index=True).rename(columns={'item_cnt_month': 'item_cnt_month_by_city_first'})

df_block = pd.merge(df_block, df_first, how='left', on=['date_block_num', 'city_code'])
# feature extraction: drop first 12 months records

df_block = df_block[df_block['date_block_num'] > 11]
# feature extraction: fillna with 0

col_fillnas = df_block.columns[df_block.isna().any()].tolist()

df_block[col_fillnas] = df_block[col_fillnas].fillna(0)
# memory clean-up

col_floats = [col for col in df_block.columns if df_block[col].dtypes == 'float64']

col_ints = [col for col in df_block.columns if df_block[col].dtypes == 'int64']

df_block[col_floats] = df_block[col_floats].astype(np.float16)

df_block[col_ints] = df_block[col_ints].astype(np.int16)

del df_first, list_of_first

gc.collect()
# describe block dataframe

df_block.describe(include='all')
# verify dtypes object

df_block.info()
# select the important features

x = df_block[(df_block['first_sale_item'] != 0) & (df_block['date_block_num'] >= 30) & (df_block['date_block_num'] <= 33)].drop(['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'item_category_id', 'year', 'month', 'day', 'city_code', 'type_code', 'subtype_code'] + [col for col in df_block.columns if col.startswith('item_price')], axis=1)

y = df_block.loc[(df_block['first_sale_item'] != 0) & (df_block['date_block_num'] >= 30) & (df_block['date_block_num'] <= 33), 'item_cnt_month']
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, random_state=58, test_size=0.25)
# memory clean-up

del x, y

gc.collect()
# xgboost regression model setup

model_xgbreg = xgb.XGBRegressor(max_depth=8, learning_rate=0.3, n_estimators=500, objective='reg:linear', booster='gbtree', gamma=0.1, min_child_weight=300, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1, random_state=58)



# xgboost regression model fit

model_xgbreg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], early_stopping_rounds=10, verbose=False, callbacks=[xgb.callback.print_evaluation(period=10)])



# xgboost regression model prediction

model_xgbreg_ypredict = model_xgbreg.predict(x_validate).clip(0 ,20)



# xgboost regression model metrics

model_xgbreg_rmse = mean_squared_error(y_validate, model_xgbreg_ypredict) ** 0.5

print('xgboost regression\n  root mean squared error: %0.4f' %model_xgbreg_rmse)
# plot the feature importances

fig, axes = plt.subplots(figsize=(150 , 20))

xgb.plot_importance(model_xgbreg, ax=axes)
# model selection

model_xgbreg_exist = model_xgbreg
# memory clean-up

del x_train, x_validate, y_train, y_validate

gc.collect()
# select the important features

x = df_block[(df_block['first_sale_item'] == 0) & (df_block['date_block_num'] >= 30) & (df_block['date_block_num'] <= 33)].drop(['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'item_category_id', 'year', 'month', 'day', 'city_code', 'type_code', 'subtype_code'] + [col for col in df_block.columns if col.startswith('item_price')], axis=1)

y = df_block.loc[(df_block['first_sale_item'] == 0) & (df_block['date_block_num'] >= 30) & (df_block['date_block_num'] <= 33), 'item_cnt_month']
# perform train-test (validate) split

x_train, x_validate, y_train, y_validate = train_test_split(x, y, test_size=0.25, random_state=58)
# memory clean-up

del x, y

gc.collect()
# xgboost regression model setup

model_xgbreg = xgb.XGBRegressor(max_depth=8, learning_rate=0.3, n_estimators=500, objective='reg:linear', booster='gbtree', gamma=0.1, min_child_weight=300, subsample=0.8, colsample_bytree=0.8, reg_alpha=0, reg_lambda=1, random_state=58)



# xgboost regression model fit

model_xgbreg.fit(x_train, y_train, eval_set=[(x_train, y_train), (x_validate, y_validate)], early_stopping_rounds=10, verbose=True)



# xgboost regression model prediction

model_xgbreg_ypredict = model_xgbreg.predict(x_validate).clip(0 ,20)



# xgboost regression model metrics

model_xgbreg_rmse = mean_squared_error(y_validate, model_xgbreg_ypredict) ** 0.5

print('xgboost regression\n  root mean squared error: %0.4f' %model_xgbreg_rmse)
# plot the feature importances

fig, axes = plt.subplots(figsize=(150 , 20))

xgb.plot_importance(model_xgbreg, ax=axes)
# model selection

model_xgbreg_first = model_xgbreg
# memory clean-up

del x_train, x_validate, y_train, y_validate

gc.collect()
# feature extraction: fix the duplicated shop id

df_test.loc[df_test['shop_id'] == 0, 'shop_id'] = 57

df_test.loc[df_test['shop_id'] == 1, 'shop_id'] = 58

df_test.loc[df_test['shop_id'] == 11, 'shop_id'] = 10
# model selection

final_model = [model_xgbreg_exist, model_xgbreg_first]



# prepare testing data and compute the observed value for model_exist

x_test = df_block[(df_block['first_sale_item'] != 0) & (df_block['date_block_num'] == 34)].drop(['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'item_category_id', 'year', 'month', 'day', 'city_code', 'type_code', 'subtype_code'] + [col for col in df_block.columns if col.startswith('item_price')], axis=1)

y_test = pd.DataFrame({'item_cnt_month': final_model[0].predict(x_test).clip(0 ,20), 'shop_id': df_block.loc[(df_block['first_sale_item'] != 0) & (df_block['date_block_num'] == 34), 'shop_id'], 'item_id': df_block.loc[(df_block['first_sale_item'] != 0) & (df_block['date_block_num'] == 34), 'item_id']}, index=x_test.index)

y_submit_exist = pd.merge(y_test, df_test[['ID', 'shop_id', 'item_id']], how='left', on=['shop_id', 'item_id'])



# prepare testing data and compute the observed value for model_first

x_test = df_block[(df_block['first_sale_item'] == 0) & (df_block['date_block_num'] == 34)].drop(['date_block_num', 'shop_id', 'item_id', 'item_cnt_month', 'item_category_id', 'year', 'month', 'day', 'city_code', 'type_code', 'subtype_code'] + [col for col in df_block.columns if col.startswith('item_price')], axis=1)

y_test = pd.DataFrame({'item_cnt_month': final_model[1].predict(x_test).clip(0 ,20), 'shop_id': df_block.loc[(df_block['first_sale_item'] == 0) & (df_block['date_block_num'] == 34), 'shop_id'], 'item_id': df_block.loc[(df_block['first_sale_item'] == 0) & (df_block['date_block_num'] == 34), 'item_id']}, index=x_test.index)

y_submit_first = pd.merge(y_test, df_test[['ID', 'shop_id', 'item_id']], how='left', on=['shop_id', 'item_id'])



# merge submission

y_submit = pd.concat([y_submit_exist, y_submit_first], ignore_index=True)
# submit the results

out = pd.DataFrame({'ID': y_submit['ID'], 'item_cnt_month': y_submit['item_cnt_month']})

out.to_csv('submission.csv', index=False)