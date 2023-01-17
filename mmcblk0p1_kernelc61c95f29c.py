import numpy as np

import pandas as pd

import os

print(os.listdir("../input"))
train = pd.read_csv('../input/sales_train.csv')

test = pd.read_csv('../input/test.csv')

sample = pd.read_csv('../input/sample_submission.csv')

items = pd.read_csv('../input/items.csv')

cats = pd.read_csv('../input/item_categories.csv')

shops = pd.read_csv('../input/shops.csv')
train.head()
test.head()
print('train',train.shape,

      'test',test.shape,

      'items',items.shape,

      'cats',cats.shape,

      'shops',shops.shape )
print('train:',train.isnull().sum().sum())

print('test:',test.isnull().sum().sum())

print('items:',items.isnull().sum().sum())

print('cats:',cats.isnull().sum().sum())

print('shops:',shops.isnull().sum().sum())
good_sales = test.merge(train, on=['item_id','shop_id'], how='left').dropna()

good_pairs = test[test['ID'].isin(good_sales['ID'])]

no_data_items = test[~(test['item_id'].isin(train['item_id']))]



print('1. Number of good pairs:', len(good_pairs))

print('2. No Data Items:', len(no_data_items))

print('3. Only Item_id Info:', len(test)-len(no_data_items)-len(good_pairs))
import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline

plt.figure(figsize=(10,4))

plt.xlim(-100, 3000)

sns.boxplot(x=train.item_cnt_day)



plt.figure(figsize=(10,4))

plt.xlim(train.item_price.min(), train.item_price.max()*1.1)

sns.boxplot(x=train.item_price)
train = train[train.item_price<100000]

train = train[train.item_cnt_day<1001]
median = train[(train.shop_id==32)&(train.item_id==2973)&(train.date_block_num==4)&(train.item_price>0)].item_price.median()

train.loc[train.item_price<0, 'item_price'] = median
from math import ceil

grouped = pd.DataFrame(train.groupby(['shop_id', 'date_block_num'])['item_cnt_day'].sum().reset_index())

fig, axes = plt.subplots(nrows=5, ncols=2, sharex=True, sharey=True, figsize=(16,20))

num_graph = 10

id_per_graph = ceil(grouped.shop_id.max() / num_graph)

count = 0

for i in range(5):

    for j in range(2):

        sns.pointplot(x='date_block_num', y='item_cnt_day', hue='shop_id', data=grouped[np.logical_and(count*id_per_graph <= grouped['shop_id'], grouped['shop_id'] < (count+1)*id_per_graph)], ax=axes[i][j])

        count += 1
# Якутск Орджоникидзе, 56

train.loc[train.shop_id == 0, 'shop_id'] = 57

test.loc[test.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train.loc[train.shop_id == 1, 'shop_id'] = 58

test.loc[test.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train.loc[train.shop_id == 10, 'shop_id'] = 11

test.loc[test.shop_id == 10, 'shop_id'] = 11
from itertools import product

from sklearn.preprocessing import LabelEncoder

shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]



cats['split'] = cats['item_category_name'].str.split('-')

cats['type'] = cats['split'].map(lambda x: x[0].strip())

cats['type_code'] = LabelEncoder().fit_transform(cats['type'])

# if subtype is nan then type

cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats = cats[['item_category_id','type_code', 'subtype_code']]



items.drop(['item_name'], axis=1, inplace=True)
len(list(set(test.item_id) - set(test.item_id).intersection(set(train.item_id)))), len(list(set(test.item_id))), len(test)
matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train[train.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)

matrix
train['revenue'] = train['item_price'] *  train['item_cnt_day']
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['item_cnt_month'] = (matrix['item_cnt_month']

                                .fillna(0)

                                .clip(0,20) # NB clip target here

                                .astype(np.float16))
test['date_block_num'] = 34

test['date_block_num'] = test['date_block_num'].astype(np.int8)

test['shop_id'] = test['shop_id'].astype(np.int8)

test['item_id'] = test['item_id'].astype(np.int16)
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True) # 34 month
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')

matrix = pd.merge(matrix, items, on=['item_id'], how='left')

matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')

matrix['city_code'] = matrix['city_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['type_code'] = matrix['type_code'].astype(np.int8)

matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
data = matrix[[

    'date_block_num',

    'shop_id',

    'item_id',

    'item_cnt_month',

    'city_code',

    'item_category_id',

    'type_code',

    'subtype_code']]
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)

Y_train = data[data.date_block_num < 33]['item_cnt_month']

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

Y_valid = data[data.date_block_num == 33]['item_cnt_month']

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
from xgboost import XGBRegressor

from xgboost import plot_importance



def plot_features(booster, figsize):    

    fig, ax = plt.subplots(1,1,figsize=figsize)

    return plot_importance(booster=booster, ax=ax)



model = XGBRegressor(

    max_depth=8,

    n_estimators=123,

    min_child_weight=400, 

    colsample_bytree=0.8, 

    subsample=0.8, 

    eta=0.3,    

    seed=42)



model.fit(

    X_train, 

    Y_train, 

    eval_metric="rmse", 

    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 

    verbose=True, 

    early_stopping_rounds = 10)
Y_test = model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.index, 

    "item_cnt_month": Y_test

})

submission.to_csv('xgb_submission.csv', index=False)
submission
# train['date'] = pd.to_datetime(train['date'],format = '%d.%m.%Y')

# dataset = train.pivot_table(index = ['shop_id','item_id'],values = ['item_cnt_day'],

#                                  columns = ['date_block_num'],fill_value = 0)
# dataset.reset_index(inplace = True)

# dataset = pd.merge(test,dataset,on = ['item_id','shop_id'],how = 'left')

# dataset.fillna(0,inplace = True)

# dataset.drop(['shop_id','item_id','ID'],inplace = True, axis = 1)

# print(dataset.head())
# X_train = np.expand_dims(dataset.values[:,:-1],axis = 2)

# y_train = dataset.values[:,-1:]

# X_test = np.expand_dims(dataset.values[:,1:],axis = 2)

# print(X_train.shape,y_train.shape,X_test.shape)
# X_train = data[data.date_block_num < 34].drop(['item_cnt_month'], axis=1)

# Y_train = data[data.date_block_num < 34]['item_cnt_month']

# X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

# print(X_train.shape,Y_train.shape,X_test.shape)
# from keras.models import Sequential

# from keras.layers import LSTM,Dense,Dropout

# from keras import optimizers

# #model 

# my_model = Sequential()

# my_model.add(LSTM(units = 99,input_shape = (7,1)))

# my_model.add(Dropout(0.5))

# my_model.add(Dense(1))



# sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# my_model.compile(loss = 'mse',optimizer = sgd, metrics = ['mse'])

# my_model.summary()
# history=my_model.fit(X_train,Y_train,batch_size = 1000,epochs = 5)
# submission= my_model.predict(X_test)

# submission= submission.clip(0,20)

# sub= pd.DataFrame({'ID':test['ID'],'item_cnt_month':submission.ravel()})

# sub.to_csv('submission.csv',index = False)