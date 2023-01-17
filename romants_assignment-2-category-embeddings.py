import pandas as pd

from sklearn.tree import DecisionTreeRegressor

from sklearn.linear_model import LinearRegression

from keras.layers import *

from keras.regularizers import l2

from keras.models import Model

from numpy import array

import matplotlib.pyplot as plt

import numpy as np

from keras import backend as K
# K.tensorflow_backend._get_available_gpus()
items_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

shops_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')

cats_df = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

test  = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')
train.head()
train.shape
shop_ids = test['shop_id'].unique()

item_ids = test['item_id'].unique()
filtered_train = train[train['shop_id'].isin(shop_ids)]

filtered_train = filtered_train[filtered_train['item_id'].isin(item_ids)]
print(filtered_train.shape)

filtered_train.head()
del filtered_train['date']

price_series = filtered_train[['item_price', 'item_id']].drop_duplicates(subset='item_id')

del filtered_train['item_price']
filtered_train.head()
train_df = filtered_train.groupby(['shop_id', 'item_id', 'date_block_num']).sum().reset_index()

train_df = train_df.rename(columns={'item_cnt_day': 'total_sales'}).sort_values(by = ['date_block_num', 'item_id', 'shop_id'])

train_df.head()
train_df.shape
clf = DecisionTreeRegressor()

clf.fit(train_df[['shop_id', 'item_id', 'date_block_num']], train_df['total_sales'])
test['date_block_num'] = [34]*len(test)

test.head()
predictions = clf.predict(test)

res_df = pd.DataFrame(predictions, columns=['item_cnt_month'])

res_df['ID'] = test.index

res_df = res_df[['ID', 'item_cnt_month']]

res_df.to_csv('solid_benchmark.csv', index=False)

res_df.head()
shops = {p:i for (i,p) in enumerate(set(np.concatenate((train_df['shop_id'].unique(), test['shop_id'].unique()))))}

items = {p:i for (i,p) in enumerate(set(np.concatenate((train_df['item_id'].unique(), test['item_id'].unique()))))}

months = {p:i for (i,p) in enumerate(np.concatenate((train_df['date_block_num'].unique() ,np.array([34]))))}
preprocess_df = train_df.copy(deep=True)

preprocess_df['shop_id'] = [shops[i] for i in preprocess_df['shop_id']]

preprocess_df['item_id'] = [items[i] for i in preprocess_df['item_id']]

preprocess_df['date_block_num'] = [months[i] for i in preprocess_df['date_block_num']]

preprocess_df.head()
len_shop = len(shops)

len_item = len(items)

len_months = len(months)

print(f'shops: {len_shop}, items: {len_item}, month: {len_months}')
shop_inp = Input(shape=(1,),dtype='int64')

item_inp = Input(shape=(1,),dtype='int64')

months_inp = Input(shape=(1,),dtype='int64')



shop_emb = Embedding(len_shop,6,input_length=1, embeddings_regularizer=l2(1e-6))(shop_inp)

item_emb = Embedding(len_item,50,input_length=1, embeddings_regularizer=l2(1e-6))(item_inp)

months_emb = Embedding(len_months,6,input_length=1, embeddings_regularizer=l2(1e-6))(months_inp)
x = concatenate([shop_emb,item_emb, months_emb])

x = Flatten()(x)

x = BatchNormalization()(x)

x = Dense(10,activation='relu')(x)

x = Dense(10,activation='relu')(x)

x = Dropout(0.4)(x)

x = BatchNormalization()(x)

x = Dense(10,activation='relu')(x)

x = Dense(10,activation='relu')(x)

x = Dropout(0.7)(x)

x = Dense(1,activation='relu')(x)

nn_model = Model([shop_inp,item_inp,months_inp],x)

nn_model.compile(loss = 'mse',optimizer='adam')
nn_model.summary()
history = nn_model.fit([preprocess_df['shop_id'], preprocess_df['item_id'], preprocess_df['date_block_num']] ,preprocess_df['total_sales'],epochs=10, validation_split=0.2)
fig, ax = plt.subplots(1,1,figsize=(12,4))

ax.plot(history.history['loss'])

ax.plot(history.history['val_loss'])

ax.set_title('Model loss')

ax.set_ylabel('Loss')

ax.set_xlabel('Epoch')

ax.legend(['Train', 'Validation'], loc='upper left')

plt.show()
predictions = nn_model.predict([[shops[i] for i in test['shop_id']], [items[i] for i in test['item_id']], [months[i] for i in test['date_block_num']]])

res_df = pd.DataFrame(predictions, columns=['item_cnt_month'])

res_df['ID'] = test.index

res_df = res_df[['ID', 'item_cnt_month']]

res_df.to_csv('model_11.csv', index=False)
train2_df = pd.merge(train_df,items_df, on='item_id' )

train2_df = pd.merge(train2_df,price_series, on='item_id')

train2_df.head()
len_cat = len(train2_df['item_category_id'].unique())

cat_inp = Input(shape=(1,),dtype='int64')

cat_emb = Embedding(len_cat,8,input_length=1, embeddings_regularizer=l2(1e-6))(cat_inp)

cats = {p:i for (i,p) in enumerate(np.concatenate((train2_df['item_category_id'].unique(), pd.merge(test, items_df, on='item_id', how = 'left')['item_category_id'].unique())))}

price_inp = Input(shape=(1,1),dtype='float32')
preprocess2_df = train2_df.copy(deep=True)

preprocess2_df['shop_id'] = [shops[i] for i in preprocess2_df['shop_id']]

preprocess2_df['item_id'] = [items[i] for i in preprocess2_df['item_id']]

preprocess2_df['date_block_num'] = [months[i] for i in preprocess2_df['date_block_num']]

preprocess2_df['item_category_id'] = [cats[i] for i in preprocess2_df['item_category_id']]

preprocess2_df.head()
x = concatenate([shop_emb,item_emb, months_emb, cat_emb, price_inp])

x = Flatten()(x)

x = BatchNormalization()(x)

x = Dense(10,activation='relu')(x)

x = Dense(10,activation='relu')(x)

x = Dropout(0.4)(x)

x = BatchNormalization()(x)

x = Dense(10,activation='relu')(x)

x = Dense(10,activation='relu')(x)

x = Dropout(0.7)(x)

x = Dense(1,activation='relu')(x)

nn_model2 = Model([shop_inp, item_inp, months_inp, cat_inp, price_inp],x)

nn_model2.compile(loss = 'mse',optimizer='adam')
nn_model2.summary()
history = nn_model2.fit([preprocess2_df['shop_id'], preprocess2_df['item_id'], preprocess2_df['date_block_num'], preprocess2_df['item_category_id'], np.expand_dims(np.expand_dims(preprocess2_df['item_price'], axis=1), axis=1)] ,preprocess2_df['total_sales'],epochs=10, validation_split=0.2)
fig, ax = plt.subplots(1,1,figsize=(12,4))

ax.plot(history.history['loss'])

ax.plot(history.history['val_loss'])

ax.set_title('Model loss')

ax.set_ylabel('Loss')

ax.set_xlabel('Epoch')

ax.legend(['Train', 'Validation'], loc='upper left')

plt.show()
test2 = pd.merge(test, items_df, on='item_id', how = 'left')

test2 = pd.merge(test2, price_series, on='item_id',how = 'left')

test2.head()
test2 = test2.fillna(test2['item_price'].mean())

test2.head()
del test2['item_name']
test2.head()
test2.shape
p_test2 = test2.copy(deep=True)

p_test2['shop_id'] = [shops[x] for x in p_test2['shop_id']]

p_test2['item_id'] = [items[x] for x in p_test2['item_id']]

p_test2['date_block_num'] = [months[x] for x in p_test2['date_block_num']]

p_test2['item_category_id'] = [cats[x] for x in p_test2['item_category_id']]

p_test2.shape
predictions = nn_model2.predict([p_test2['shop_id'], p_test2['item_id'], p_test2['date_block_num'], p_test2['item_category_id'], np.expand_dims(np.expand_dims(p_test2['item_price'], axis=1), axis=1)])

res_df = pd.DataFrame(predictions, columns=['item_cnt_month'])

res_df['ID'] = test2.index

res_df = res_df[['ID', 'item_cnt_month']]

res_df.to_csv('model_2.csv', index=False)
feature_extractor = Model(inputs=nn_model.input,outputs=nn_model.layers[6].output)

train_features = feature_extractor.predict([preprocess_df['shop_id'], preprocess_df['item_id'], preprocess_df['date_block_num']])

test_features = feature_extractor.predict([[shops[i] for i in test['shop_id']], [items[i] for i in test['item_id']], [months[i] for i in test['date_block_num']]])

train_features = np.squeeze(train_features) # remove dimension of 1

test_features = np.squeeze(test_features)
reg = LinearRegression().fit(train_features, preprocess_df['total_sales'])

pred = reg.predict(test_features)
res_df = pd.DataFrame(pred, columns=['item_cnt_month'])

res_df['ID'] = test2.index

res_df = res_df[['ID', 'item_cnt_month']]

res_df.to_csv('model_3.csv', index=False)