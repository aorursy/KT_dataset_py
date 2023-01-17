# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import itertools

# Input data files are available in the "../input/" directory.

# For preprocessing

from sklearn import preprocessing



import os

print(os.listdir("../input"))

print("pandas: {}".format(pd.__version__))

print('numpy: {}'.format(np.__version__))

# Any results you write to the current directory are saved as output.
shops_df = pd.read_csv('../input/shops.csv', dtype={'shop_name': 'str', 'shop_id': 'int32'})  

                            # columns: ['shop_name','shop_id']

item_cat_df = pd.read_csv('../input/item_categories.csv', 

                              dtype={'item_category_name': 'str', 'item_category_id': 'int32'}) 

                            # columns: ['item_category_name', 'item_category_id']

sales_train_df  = pd.read_csv('../input/sales_train.csv',parse_dates=['date'], 

                    dtype={'date': 'str', 'date_block_num': 'int32', 'shop_id': 'int32', 

                          'item_id': 'int32', 'item_price': 'float32', 'item_cnt_day': 'int32'}) 

                            # ['date','date_block_num','shop_id','item_id','item_price', 'item_cnt_day']

items_df = pd.read_csv('../input/items.csv', dtype={'item_name': 'str', 'item_id': 'int32', 

                                                 'item_category_id': 'int32'})  

                            # ['item_name','item_id','item_category_id']

sample_sub_df = pd.read_csv('../input/sample_submission.csv')  # ['ID',item_cnt_month']

test_df = pd.read_csv('../input/test.csv',dtype={'ID': 'int32', 'shop_id': 'int32', 'item_id': 'int32'}) # [shop_id','item_id']
# Unique ['shop_id','item_id'] pairs in train dataset

train_pairs = sales_train_df[['shop_id','item_id']].sort_values(['shop_id','item_id']).drop_duplicates().reset_index(drop=True)



# Unique ['shop_id','item_id'] pairs in train dataset

test_pairs = test_df[['shop_id','item_id']].sort_values(['shop_id','item_id']).drop_duplicates().reset_index(drop=True)

print("Number of train unique pairs: {}".format(train_pairs.shape))

print("Number of test unique pairs: {}".format(test_pairs.shape))
# Create the pairs for train set

train_pairs['shop_item'] = pd.Series([(train_pairs['shop_id'][i],train_pairs['item_id'][i]) for i in train_pairs.index])



# and for test set:

test_pairs['shop_item'] = pd.Series([(test_pairs['shop_id'][i],test_pairs['item_id'][i]) for i in test_pairs.index])



# we can actually drop the first two columns and we're left with the pair index

train_pairs.drop(['shop_id','item_id'], axis = 1, inplace = True)

test_pairs.drop(['shop_id','item_id'], axis = 1, inplace = True)
good_pairs = train_pairs.merge(test_pairs)

train_not_test =  train_pairs[~train_pairs['shop_item'].isin(test_pairs['shop_item'])]

test_not_train = test_pairs[~test_pairs['shop_item'].isin(train_pairs['shop_item'])]

## Test:

test_not_train.isin(train_not_test)['shop_item'].any()
print('------------ (shop_id,item_id) ---------')

print('unique pairs in test: {}'.format(len(test_pairs)))

print('unique pairs in train: {}'.format(len(train_pairs)))

print('good_pairs: {}'.format(len(good_pairs)))

print('train_not_test: {}'.format(len(train_not_test)))

print('test_not_train: {}'.format(len(test_not_train)))

print('ratio of test not in train: {}'.format(len(test_not_train)/len(test_pairs)))

shop_unique = pd.DataFrame(sorted(sales_train_df['shop_id'].unique()))

item_unique = pd.DataFrame(sorted(sales_train_df['item_id'].unique()))

missing_items = test_df[~test_df['item_id'].isin(sales_train_df['item_id'].unique())]['item_id'].unique()

all_items = list(set(sales_train_df['item_id']).union(missing_items))
print('number of items in train: {}'.format(len(item_unique)))

print('number of missing items: {}'.format(len(missing_items)))

print('number of total items: {}'.format(len(all_items)))

sales_by_shop = sales_train_df.pivot_table(index = ['shop_id'], values = ['item_cnt_day'], columns = ['date_block_num'], 

                                           aggfunc=np.sum, fill_value = 0).reset_index()

sales_by_shop.columns = sales_by_shop.columns.droplevel().map(str)

sales_by_shop = sales_by_shop.reset_index(drop=True).rename_axis(None, axis=1)

sales_by_shop.columns.values[0] = 'shop_id'

sales_by_shop.head()
# There are 60 shops, let's plot them side by side

fig, ax = plt.subplots(5,2, figsize=(30, 15))

ax[0,0].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[:6,1:].T, 'o-' )

ax[1,0].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[6:12,1:].T, 'o-' )

ax[2,0].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[12:18,1:].T, 'o-')

ax[3,0].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[18:24,1:].T, 'o-')

ax[4,0].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[24:30,1:].T, 'o-')

ax[0,1].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[30:36,1:].T, 'o-')

ax[1,1].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[36:42,1:].T, 'o-')

ax[2,1].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[42:48,1:].T, 'o-')

ax[3,1].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[48:54,1:].T, 'o-')

ax[4,1].plot(sales_by_shop.columns[1:], sales_by_shop.iloc[54:,1:].T, 'o-')

fig.suptitle('Montlhy sales per shop_id', fontsize = 14)  

plt.show()
plt.figure(figsize=(14,7))

plt.errorbar(sales_by_shop.columns[1:], sales_by_shop.iloc[:,1:].mean(), 

             yerr=sales_by_shop.iloc[:,1:].std()/sales_by_shop.iloc[:,1:].count().add(-1).pow(0.5), fmt='-o', ecolor='orangered',capsize=3)

plt.title('Average Monthly Sales', fontsize=14)

plt.xlabel('Month', fontsize= 12 )

plt.ylabel('Sales', fontsize = 12)

plt.show()
new = item_cat_df['item_category_name'].str.split(' - ', expand =True)

new[1]=new[1].fillna('none')

#Label encoding these two new columns

le1 = preprocessing.LabelEncoder()

le1.fit(new[0].unique())

new[2] = le1.transform(new[0])

le2 = preprocessing.LabelEncoder()

le2.fit(new[1].unique())

new[3] = le2.transform(new[1])



# Create a copy and fill it with the new columns

item_cat_exp = item_cat_df.copy() 

item_cat_exp['cat_type'] = new[0]

item_cat_exp['item_type'] = new[1]

item_cat_exp['cat_type_l'] = new[2]

item_cat_exp['item_type_l'] = new[3]

item_cat_exp.drop('item_category_name', axis =1, inplace= True)

item_cat_exp.head()
item_cat_exp.shape
sales_cat = sales_train_df.join(items_df, on = 'item_id', rsuffix='_').join(item_cat_exp, on = 'item_category_id', rsuffix = "_").drop(['item_name','item_id_','item_category_id_'], axis =1)

sales_cat.head().T
g_sales_by_cat = sales_cat.sort_values('date').groupby(['date_block_num', 'cat_type', 'cat_type_l', 'item_category_id','item_id'], as_index = False).agg({'item_price': ['mean','min','max'], 'item_cnt_day': ['sum', 'mean']})

g_sales_by_cat.head(10).T
# Let's pivot the above dataset to get a monthly time series:

sales_by_cat = sales_cat.pivot_table(index=['item_category_id'], values = ['item_cnt_day'], columns=['date_block_num'], 

                     aggfunc = np.sum, fill_value = 0 ).reset_index()

sales_by_cat.columns = sales_by_cat.columns.droplevel().map(str)

sales_by_cat = sales_by_cat.reset_index(drop=True).rename_axis(None, axis=1)

sales_by_cat.columns.values[0]= 'item_category_id'

sales_by_cat.head()
item_cat_df.item_category_id.unique()
fig, ax = plt.subplots(6,2,figsize = (30,15))

ax[0][0].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[:8,1:].T, 'o-')

ax[1][0].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[8:16,1:].T, 'o-')

ax[2][0].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[16:24,1:].T, 'o-')

ax[3][0].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[24:32,1:].T, 'o-')

ax[4][0].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[32:40,1:].T, 'o-')

ax[5][0].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[40:48,1:].T, 'o-')

ax[0][1].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[48:56,1:].T, 'o-')

ax[1][1].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[56:64,1:].T, 'o-')

ax[2][1].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[64:72,1:].T, 'o-')

ax[3][1].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[72:80,1:].T, 'o-')

ax[4][1].plot(sales_by_cat.columns[1:], sales_by_cat.iloc[80:,1:].T, 'o-')



plt.show()
price_by_cat = sales_cat.pivot_table(index=['item_category_id'], values = ['item_price'], columns=['date_block_num'], 

                     aggfunc = np.mean, fill_value = 0 ).reset_index()

price_by_cat.columns = price_by_cat.columns.droplevel().map(str)

price_by_cat = price_by_cat.reset_index(drop=True).rename_axis(None, axis=1)

price_by_cat.columns.values[0]= 'item_category_id'

price_by_cat.head()
fig, ax = plt.subplots(6,2,figsize = (30,15))

ax[0][0].plot(price_by_cat.columns[1:], price_by_cat.iloc[:8,1:].T, 'o-')

ax[1][0].plot(price_by_cat.columns[1:], price_by_cat.iloc[8:16,1:].T, 'o-')

ax[2][0].plot(price_by_cat.columns[1:], price_by_cat.iloc[16:24,1:].T, 'o-')

ax[3][0].plot(price_by_cat.columns[1:], price_by_cat.iloc[24:32,1:].T, 'o-')

ax[4][0].plot(price_by_cat.columns[1:], price_by_cat.iloc[32:40,1:].T, 'o-')

ax[5][0].plot(price_by_cat.columns[1:], price_by_cat.iloc[40:48,1:].T, 'o-')

ax[0][1].plot(price_by_cat.columns[1:], price_by_cat.iloc[48:56,1:].T, 'o-')

ax[1][1].plot(price_by_cat.columns[1:], price_by_cat.iloc[56:64,1:].T, 'o-')

ax[2][1].plot(price_by_cat.columns[1:], price_by_cat.iloc[64:72,1:].T, 'o-')

ax[3][1].plot(price_by_cat.columns[1:], price_by_cat.iloc[72:80,1:].T, 'o-')

ax[4][1].plot(price_by_cat.columns[1:], price_by_cat.iloc[80:,1:].T, 'o-')



plt.show()
avg_sales_by_cat = sales_cat.pivot_table(index=['item_category_id'], values = ['item_cnt_day'], columns=['date_block_num'], 

                     aggfunc = np.mean, fill_value = 0 ).reset_index()

avg_sales_by_cat.columns = avg_sales_by_cat.columns.droplevel().map(str)

avg_sales_by_cat = avg_sales_by_cat.reset_index(drop=True).rename_axis(None, axis=1)

avg_sales_by_cat.columns.values[0]= 'item_category_id'

avg_sales_by_cat.head()
fig, ax = plt.subplots(6,2,figsize = (30,15))

ax[0][0].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[:8,1:].T, 'o-')

ax[1][0].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[8:16,1:].T, 'o-')

ax[2][0].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[16:24,1:].T, 'o-')

ax[3][0].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[24:32,1:].T, 'o-')

ax[4][0].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[32:40,1:].T, 'o-')

ax[5][0].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[40:48,1:].T, 'o-')

ax[0][1].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[48:56,1:].T, 'o-')

ax[1][1].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[56:64,1:].T, 'o-')

ax[2][1].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[64:72,1:].T, 'o-')

ax[3][1].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[72:80,1:].T, 'o-')

ax[4][1].plot(avg_sales_by_cat.columns[1:], avg_sales_by_cat.iloc[80:,1:].T, 'o-')



plt.show()
cat_price_cnt = sales_cat.sort_values('date').groupby(['date_block_num','item_category_id']).agg({'item_price': 'mean', 'item_cnt_day': ['sum', 'mean']}).reset_index()

cat_price_cnt.columns = cat_price_cnt.columns.droplevel().map(str)

cat_price_cnt.columns = ['date_block_num', 'item_category_id', 'item_price_mean','item_cnt_month', 'item_cnt_mean' ]

cat_price_cnt.head()
sales_cat.head().T
cat_type_price_cnt = sales_cat.sort_values('date').groupby(['date_block_num','cat_type','cat_type_l']).agg({'item_price': 'mean', 'item_cnt_day': ['sum', 'mean']}).reset_index()

cat_type_price_cnt.columns = cat_type_price_cnt.columns.droplevel().map(str)

cat_type_price_cnt.columns = ['date_block_num', 'cat_type', 'cat_type_l', 'item_price_mean','item_cnt_month', 'item_cnt_mean' ]

cat_type_price_cnt.head()
## This snippet has to be transformed into a function 



item_type_price_cnt = sales_cat.sort_values('date').groupby(['date_block_num','item_type','item_type_l']).agg({'item_price': 'mean', 'item_cnt_day': ['sum', 'mean']}).reset_index()

item_type_price_cnt.columns = item_type_price_cnt.columns.droplevel().map(str)

item_type_price_cnt.columns = ['date_block_num', 'item_type', 'item_type_l', 'item_price_mean','item_cnt_month', 'item_cnt_mean' ]

item_type_price_cnt.head()
sales_train_df.sort_values('date').groupby(['date_block_num', 'item_id']).agg({'item_price': ['mean', 'std'], 'item_cnt_day':['sum', 'mean', 'std']}).reset_index().head().T
fig, ax = plt.subplots(2,1,figsize=(10,6))

# plt.figure(figsize=(10,4))

# plt.xlim(-100, 3000)

sns.boxplot(x=sales_train_df['item_cnt_day'], ax =ax[0], palette='Set3' ).set_title('item_cnt_day distribution')

# plt.figure(figsize=(10,4))

# plt.xlim(sales_train_df['item_price'].min(), sales_train_df['item_price'].max())

sns.boxplot(x=sales_train_df['item_price'], ax = ax[1] ).set_title('item_price distribution')

plt.tight_layout()

plt.show()
# There are repeated shops with different id let's correct this

# Якутск Орджоникидзе, 56

sales_train_df.loc[sales_train_df.shop_id == 0, 'shop_id'] = 57

test_df.loc[test_df.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

sales_train_df.loc[sales_train_df.shop_id == 1, 'shop_id'] = 58

test_df.loc[test_df.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

sales_train_df.loc[sales_train_df.shop_id == 10, 'shop_id'] = 11

test_df.loc[test_df.shop_id == 10, 'shop_id'] = 11
# print('Item outliers by sales:',sales_train_df['item_id'][sales_train_df['item_cnt_day']>400].unique())

# print('Item outliers by price:',sales_train_df['item_id'][sales_train_df['item_price']>40000].unique())

train = sales_train_df.join(items_df, on='item_id', rsuffix='_').join(shops_df, on='shop_id', rsuffix='_').join(item_cat_df, on='item_category_id', rsuffix='_').drop(['item_id_', 'shop_id_', 'item_category_id_'], axis=1)

train.head().T
# Observe that there are no null values, this will change when we merge with the test_df

#train.isna().sum()

train.isnull().sum()
print('train shape: {}'.format(train.shape))

print('time period:\n\t start -> {} \n\t  end -> {} '.format(train['date'].min().date(), train['date'].max().date()))
sales_by_item_id = sales_train_df.pivot_table(index=['item_id'],values=['item_cnt_day'], 

                                        columns='date_block_num', aggfunc=np.sum, fill_value=0).reset_index()

sales_by_item_id.columns = sales_by_item_id.columns.droplevel().map(str)

sales_by_item_id = sales_by_item_id.reset_index(drop=True).rename_axis(None, axis=1)

sales_by_item_id.columns.values[0] = 'item_id'
outdated_items = sales_by_item_id[sales_by_item_id.loc[:,'27':].sum(axis=1) == 0]
sales_by_item_id[sales_by_item_id.loc[:,'27':].sum(axis=1) == 0].shape[0]/sales_by_item_id.shape[0]
outdated_test = test_df[test_df['item_id'].isin(outdated_items['item_id'])]
test_df[test_df['item_id'].isin(outdated_items['item_id'])].shape[0]/test_df.shape[0]
train = train.query('item_price > 0 and item_price < 50000')

train = train.query('item_cnt_day <= 20')

train.shape
train[train['item_category_id'] == 4].head().T
train = train.drop(['item_name', 'shop_name', 'item_category_name'], axis = 1)

train.head().T
train_month = train.sort_values('date').groupby(['date_block_num','shop_id', 'item_category_id', 'item_id'], as_index = False).agg({'item_price': 'mean', 'item_cnt_day':['sum', 'mean','count']})

train_month.columns = ['date_block_num', 'shop_id', 'item_category_id', 'item_id', 'mean_item_price', 'item_cnt', 'mean_item_cnt', 'transactions']

train_month.head(10).T
train_month.describe()
shop_unique = pd.DataFrame(sorted(train_month['shop_id'].unique()))

item_unique = pd.DataFrame(sorted(train_month['item_id'].unique()))
train_month['shop_id'].unique()
# Merge the train set with the complete set (missing records will be filled with 0).

train_month = pd.merge(blank_df, train_month, on=['date_block_num','shop_id','item_id'], how='left')

train_month.fillna(0, inplace=True)
# we need to fill this database more carefully, at least we should retrieve the item_category_id for each pair. The above code is just filling in 0 for every NaN value

# this cannot be good for the model, but let's continue for the purpose of having a first complete pipeline .

train_month.head().T