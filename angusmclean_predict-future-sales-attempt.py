# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
SET_DIVISOR = 100
sales_train = pd.read_csv('../input/sales_train.csv')[::SET_DIVISOR]
items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
item_categories = pd.read_csv('../input/item_categories.csv')
sales_train.head()
item_categories.head()
item_categories.iloc[1]['item_category_name'].split(' - ')[0]
pd.read_csv('../input/sample_submission.csv')
# Fix dates
import datetime

sales_train.date = sales_train.date.apply(lambda x:datetime.datetime.strptime(x, '%d.%m.%Y'))
sales_train['month'] = sales_train.date.apply(lambda x: x.strftime('%m'))
sales_train['year'] = sales_train.date.apply(lambda x: x.strftime('%Y'))
# Add item categories to rows
sales_train = sales_train.merge(items, on=['item_id'])
item_categories['super_cat_name'] = item_categories.item_category_name.apply(lambda x: x.split(' - ')[0])
item_categories
sales_train = sales_train.merge(item_categories[['item_category_id', 'super_cat_name']], on=['item_category_id'])
sales_train.head()
sales_train.groupby(['item_id']).agg({
    "item_name" : "first"
})
numShopsForItem = sales_train.groupby(['item_id', 'shop_id']).agg({'date_block_num' : ['min', 'max']}).reset_index().groupby(['item_id']).agg({'shop_id':{'Num Shops':'count'}})

#numShopsForItem.size
numStoresAvailAt, freq = np.unique(numShopsForItem['shop_id']['Num Shops'], return_counts=True)
plt.scatter(numStoresAvailAt, freq)
plt.xlabel('Number of Stores')
plt.ylabel('Frequency')
plt.title("Typical Availability of Items")
numDatesForItem = sales_train.groupby(['item_id', 'date_block_num']).agg({'shop_id' : 'count'}).reset_index().groupby(['item_id']).agg({'date_block_num':'count'})

numStoresAvailAt, freq = np.unique(numDatesForItem['date_block_num'], return_counts=True)
plt.scatter(numStoresAvailAt, freq)
plt.xlabel('Number of Months')
plt.ylabel('Frequency')
plt.title("Typical Lifetime of Items")
# Monthly sales for all items in all shops
monthlySales = sales_train.groupby(['date_block_num'])['item_cnt_day'].sum().reset_index()
plt.plot(monthlySales.date_block_num, monthlySales.item_cnt_day)
# Monthly sales for each category
catMonthlyCounts = sales_train.groupby(['item_category_id', 'date_block_num'])['item_cnt_day'].sum().reset_index()

for catId in catMonthlyCounts.item_category_id.unique():
  tmp = catMonthlyCounts[catMonthlyCounts.item_category_id == catId]
  plt.plot(tmp.date_block_num, tmp.item_cnt_day)
# Changing prices for items over time
catMonthlyCounts = sales_train.groupby(['item_id', 'date_block_num'])['item_price'].mean().reset_index()

for itemId in catMonthlyCounts.item_id.unique()[::99]:
  tmp = catMonthlyCounts[catMonthlyCounts.item_id == itemId]
  plt.plot(tmp.date_block_num, tmp.item_price)
# Monthly sales for each category
superCatMonthlyCounts = sales_train.groupby(['super_cat_name', 'date_block_num'])['item_cnt_day'].sum().reset_index()

for catId in superCatMonthlyCounts.super_cat_name.unique():
  tmp = superCatMonthlyCounts[superCatMonthlyCounts.super_cat_name == catId]
  plt.plot(tmp.date_block_num, tmp.item_cnt_day)
# Monthly sales for each shop
shopMonthlyCounts = sales_train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index()

for shopId in shopMonthlyCounts.shop_id.unique():
  tmp = shopMonthlyCounts[shopMonthlyCounts.shop_id == shopId]
  plt.plot(tmp.date_block_num, tmp.item_cnt_day)
# sales_train.columns
# def most_common(arr):
#     a = arr.value_counts()
#     return a.axes[0]
# mnthly_sales = sales_train.groupby(['item_id', 'shop_id', 'date_block_num']).agg({
#     'item_price':'mean',
#     'item_cnt_day': 'sum',
#     'month': most_common, 'year': most_common, 'item_category_id': most_common, 'super_cat_name': most_common
# })
# mnthly_sales
# reind = np.array(np.meshgrid(
#     sales_train['item_id'].unique(),
#     sales_train['shop_id'].unique(),
#     sales_train['date_block_num'].unique()
# )).T.reshape(-1,3)
# reind = [tuple(row) for row in reind]
# print(len(sales_train['item_id'].unique()))
# print(len(sales_train['shop_id'].unique()))
# print(len(sales_train['date_block_num'].unique()))
# print(len(reind))
# index = pd.MultiIndex.from_tuples(reind, names=['item_id', 'shop_id', 'date_block_num'])
# mnthly_sales = mnthly_sales.reindex(index)
# mnthly_sales.loc[2757]
# NUM_DATE_BLOCKS = len(sales_train.date_block_num.unique())
# NUM_ITEMS = len(sales_train.item_id.unique())
# NUM_SHOPS = len(sales_train.shop_id.unique())

# cols = ["item_id", "date_block_num", "shop_id", "item_price", "item_cnt_day"]

# print(NUM_DATE_BLOCKS, NUM_ITEMS, NUM_SHOPS, ' -- Total Rows :', NUM_DATE_BLOCKS * NUM_ITEMS * NUM_SHOPS)
# date_nums = sales_train.date_block_num.unique()
# items = sales_train.item_id.unique()[:10]

# shape = (len(date_nums) * len(items) * NUM_SHOPS, len(cols))
# data = np.zeros(shape)

# rowNum = 0
# for item in items:
#     avgPrice = mnthly_sales.loc[(item)].item_price.mean()
#     for shop in sales_train.shop_id.unique():
#         for date_num in date_nums:
            

#             if (item, shop, date_num) not in mnthly_sales.index:
#                 data[rowNum][3] = avgPrice
#                 data[rowNum][4] = 0
#             else: 
#                 data[rowNum][3] = mnthly_sales.loc[(item, shop, date_num)]["item_price"]
#                 data[rowNum][4] = mnthly_sales.loc[(item, shop, date_num)]["item_cnt_day"]
#             rowNum += 1
from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler, OneHotEncoder, LabelBinarizer
from keras.callbacks import LambdaCallback
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, LSTM, Input, Embedding, Flatten, Dropout, merge
from keras.optimizers import RMSprop
sales_train.columns
cont_cols = ["item_price"]
cat_cols = ["date_block_num", "shop_id", 'month', 'year', "item_category_id", "super_cat_name"]
onehot = sales_train.copy()
for o in cat_cols:
    print('Vectorizing ', o,'...')
    onehot[o] = LabelBinarizer().fit_transform(sales_train[o]).tolist()
cat_data = onehot[cat_cols].as_matrix()
cat_data = onehot[cat_cols].values
for ind,row in enumerate(cat_data):
    cat_data[ind] = np.array(
        [np.array(feat) for feat in row]
    )
cat_maps = [(o, LabelEncoder()) for o in cat_cols]
cat_mapper = DataFrameMapper(cat_maps).fit(sales_train)
cat_data2 = cat_mapper.transform(sales_train)
cont_maps = [([o], StandardScaler()) for o in cont_cols]
cont_mapper = DataFrameMapper(cont_maps).fit(sales_train)
cont_data = cont_mapper.transform(sales_train)
print(cat_data2.reshape(29359, 6, -1).shape)
print(cont_data.shape)
vectorized = np.concatenate((cat_data2.reshape(29359, 6), cont_data.reshape(29359, 1)), axis=1)
print(vectorized.shape)
print(vectorized[0])
def buildEmbedding(name, num_cat):
    c2 = (num_cat+1)//2
    if c2>50: c2=50
    inp = Input((1,), dtype='int64', name=name+'_in')
    # , W_regularizer=l2(1e-6)
    u = Flatten(name=name+'_flt')(Embedding(num_cat, c2, input_length=1)(inp))
    return inp, u
# Build input layers for different features
# Continuous features
contin_inp = Input((len(cont_cols),), name='contin')
contin_out = Dense(len(cont_cols)*10, activation='relu', name='contin_d')(contin_inp)

# Categorical features
embs = [buildEmbedding(col, len(sales_train[col].unique())) for (ind, col) in enumerate(cat_cols)]
x = merge([emb for inp,emb in embs] + [contin_out], mode='concat')

x = Dropout(0.02)(x)
x = Dense(1000, activation='relu', init='uniform')(x)
x = Dense(500, activation='relu', init='uniform')(x)
x = Dropout(0.2)(x)
x = Dense(1, activation='sigmoid')(x)

model = Model([inp for inp,emb in embs] + [contin_inp], x)
model.compile('sgd', 'mean_squared_error')
def split_cols(arr): return np.hsplit(arr,arr.shape[1])
# input_array = train_x[:,[1,2]].reshape(len(train_x), 2)
# input_array = [np.array(row) for row in input_array]
# input_array = np.array(input_array)
# #input_array = input_array.reshape(len(input_array), -1, 2)
# input_array = split_cols(input_array)
# num_cat = 80

# embs = [buildEmbedding('a', num_cat), buildEmbedding('b', num_cat)]
# x = merge([emb for inp,emb in embs], mode='concat')

# tmpModel = Model([inp for inp,emb in embs], x)
# tmpModel.compile('rmsprop', 'mse')
# output_array = tmpModel.predict(input_array)
split = int(len(vectorized)*0.8)

train_x = split_cols(vectorized[:split])
train_y = sales_train["item_cnt_day"].values[:split]
val_x = split_cols(vectorized[split:])
val_y = sales_train["item_cnt_day"].values[split:]

model.fit(train_x, train_y, nb_epoch=25, validation_data=(val_x, val_y))
val_y_pred = model.predict(split_cols(vectorized[split:]))
np.sqrt(np.mean((val_y_pred-val_y)**2))