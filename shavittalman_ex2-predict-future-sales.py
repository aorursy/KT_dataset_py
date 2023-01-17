import numpy as np
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)

from itertools import product
from sklearn.preprocessing import LabelEncoder

import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

from xgboost import XGBRegressor
from xgboost import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

import time
import sys
import gc
import pickle
sys.version_info
items = pd.read_csv('../input/items.csv')
shops = pd.read_csv('../input/shops.csv')
cats = pd.read_csv('../input/item_categories.csv')
train = pd.read_csv('../input/sales_train.csv')
test  = pd.read_csv('../input/test.csv').set_index('ID')
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
# Якутск Орджоникидзе, 56
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11
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
ts = time.time()
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
time.time() - ts
train['revenue'] = train['item_price'] *  train['item_cnt_day']
ts = time.time()
group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20) # NB clip target here
                                .astype(np.float16))
time.time() - ts
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)
ts = time.time()
matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month
time.time() - ts
ts = time.time()
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['city_code'] = matrix['city_code'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
time.time() - ts
matrix.head()
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
ts = time.time()
matrix = lag_feature(matrix, [1,2,3,6,12], 'item_cnt_month')
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num'], how='left')
matrix['date_avg_item_cnt'] = matrix['date_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_avg_item_cnt')
matrix.drop(['date_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_cnt'] = matrix['date_item_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_item_avg_item_cnt')
matrix.drop(['date_item_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_shop_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_avg_item_cnt'] = matrix['date_shop_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1,2,3,6,12], 'date_shop_avg_item_cnt')
matrix.drop(['date_shop_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_cat_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_category_id'], how='left')
matrix['date_cat_avg_item_cnt'] = matrix['date_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_cat_avg_item_cnt')
matrix.drop(['date_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'item_category_id']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_cat_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'item_category_id'], how='left')
matrix['date_shop_cat_avg_item_cnt'] = matrix['date_shop_cat_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_cat_avg_item_cnt')
matrix.drop(['date_shop_cat_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_type_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'type_code'], how='left')
matrix['date_shop_type_avg_item_cnt'] = matrix['date_shop_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_type_avg_item_cnt')
matrix.drop(['date_shop_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'shop_id', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = ['date_shop_subtype_avg_item_cnt']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'shop_id', 'subtype_code'], how='left')
matrix['date_shop_subtype_avg_item_cnt'] = matrix['date_shop_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_shop_subtype_avg_item_cnt')
matrix.drop(['date_shop_subtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'city_code'], how='left')
matrix['date_city_avg_item_cnt'] = matrix['date_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_city_avg_item_cnt')
matrix.drop(['date_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'item_id', 'city_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_item_city_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'item_id', 'city_code'], how='left')
matrix['date_item_city_avg_item_cnt'] = matrix['date_item_city_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_item_city_avg_item_cnt')
matrix.drop(['date_item_city_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_subtype_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')
matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')
matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = train.groupby(['item_id']).agg({'item_price': ['mean']})
group.columns = ['item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['item_id'], how='left')
matrix['item_avg_item_price'] = matrix['item_avg_item_price'].astype(np.float16)

group = train.groupby(['date_block_num','item_id']).agg({'item_price': ['mean']})
group.columns = ['date_item_avg_item_price']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','item_id'], how='left')
matrix['date_item_avg_item_price'] = matrix['date_item_avg_item_price'].astype(np.float16)

lags = [1,2,3,4,5,6]
matrix = lag_feature(matrix, lags, 'date_item_avg_item_price')

for i in lags:
    matrix['delta_price_lag_'+str(i)] = \
        (matrix['date_item_avg_item_price_lag_'+str(i)] - matrix['item_avg_item_price']) / matrix['item_avg_item_price']

def select_trend(row):
    for i in lags:
        if row['delta_price_lag_'+str(i)]:
            return row['delta_price_lag_'+str(i)]
    return 0
    
matrix['delta_price_lag'] = matrix.apply(select_trend, axis=1)
matrix['delta_price_lag'] = matrix['delta_price_lag'].astype(np.float16)
matrix['delta_price_lag'].fillna(0, inplace=True)

# https://stackoverflow.com/questions/31828240/first-non-null-value-per-row-from-a-list-of-pandas-columns/31828559
# matrix['price_trend'] = matrix[['delta_price_lag_1','delta_price_lag_2','delta_price_lag_3']].bfill(axis=1).iloc[:, 0]
# Invalid dtype for backfill_2d [float16]

fetures_to_drop = ['item_avg_item_price', 'date_item_avg_item_price']
for i in lags:
    fetures_to_drop += ['date_item_avg_item_price_lag_'+str(i)]
    fetures_to_drop += ['delta_price_lag_'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)

time.time() - ts
ts = time.time()
group = train.groupby(['date_block_num','shop_id']).agg({'revenue': ['sum']})
group.columns = ['date_shop_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num','shop_id'], how='left')
matrix['date_shop_revenue'] = matrix['date_shop_revenue'].astype(np.float32)

group = group.groupby(['shop_id']).agg({'date_shop_revenue': ['mean']})
group.columns = ['shop_avg_revenue']
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['shop_id'], how='left')
matrix['shop_avg_revenue'] = matrix['shop_avg_revenue'].astype(np.float32)

matrix['delta_revenue'] = (matrix['date_shop_revenue'] - matrix['shop_avg_revenue']) / matrix['shop_avg_revenue']
matrix['delta_revenue'] = matrix['delta_revenue'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'delta_revenue')

matrix.drop(['date_shop_revenue','shop_avg_revenue','delta_revenue'], axis=1, inplace=True)
time.time() - ts
ts = time.time()
group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})
group.columns = [ 'date_type_avg_item_cnt' ]
group.reset_index(inplace=True)

matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')
matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)
matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')
matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)
time.time() - ts
matrix['month'] = matrix['date_block_num'] % 12
days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
matrix['days'] = matrix['month'].map(days).astype(np.int8)

ts = time.time()
cache = {}
matrix['item_shop_last_sale'] = -1
matrix['item_shop_last_sale'] = matrix['item_shop_last_sale'].astype(np.int8)

for row in matrix.itertuples():
    idx = getattr(row,'Index')
    item_id = getattr(row,'item_id')
    shop_id = getattr(row,'shop_id')
    date_block_num = getattr(row,'date_block_num')
    item_cnt_month = getattr(row,'item_cnt_month')
    key = str(item_id)+' '+str(shop_id)
    if key not in cache:
        if item_cnt_month!=0:
            cache[key] = date_block_num
    else:
        last_date_block_num = cache[key]
        matrix.at[idx, 'item_shop_last_sale'] = date_block_num - last_date_block_num
        cache[key] = date_block_num

time.time() - ts
        
 

ts = time.time()
cache = {}
matrix['item_last_sale'] = -1
matrix['item_last_sale'] = matrix['item_last_sale'].astype(np.int8)
for idx, row in matrix.iterrows():    
    key = row.item_id
    if key not in cache:
        if row.item_cnt_month!=0:
            cache[key] = row.date_block_num
    else:
        last_date_block_num = cache[key]
        if row.date_block_num>last_date_block_num:
            matrix.at[idx, 'item_last_sale'] = row.date_block_num - last_date_block_num
            cache[key] = row.date_block_num         
time.time() - ts
ts = time.time()
matrix['item_shop_first_sale'] = matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
matrix['item_first_sale'] = matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
time.time() - ts
ts = time.time()
matrix = matrix[matrix.date_block_num > 11]
time.time() - ts
ts = time.time()
def fill_na(df):
    for col in df.columns:
        if ('_lag_' in col) & (df[col].isnull().any()):
            if ('item_cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)
time.time() - ts
matrix.columns
matrix.info()
matrix.to_pickle('data.pkl')
del matrix
del cache
del group
del items
del shops
del cats
del train
# leave test for submission
gc.collect();
data = pd.read_pickle('data.pkl')
# Benchmark of XGBoost - only the basic parameters 
data1 = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month'
]]

X_train1 = data1[data1.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train1 = data1[data1.date_block_num < 33]['item_cnt_month']
X_valid1 = data1[data1.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid1 = data1[data1.date_block_num == 33]['item_cnt_month']
X_test1 = data1[data1.date_block_num == 34].drop(['item_cnt_month'], axis=1)

model1 = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)
model1.fit(
    X_train1, 
    Y_train1, 
    eval_metric="rmse", 
    eval_set=[(X_train1, Y_train1), (X_valid1, Y_valid1)], 
    verbose=True, 
    early_stopping_rounds = 5)
Y_pred1 = model1.predict(X_valid1).clip(0, 20)
Y_test1 = model1.predict(X_test1).clip(0, 20)


# save model and predictions (for future, if needed)
pickle.dump(Y_pred1, open('benchmark_pred1.pickle', 'wb'))
pickle.dump(Y_test1, open('benchmark_test1.pickle', 'wb'))
Y_pred1[:10]
Y_test1[:10]
del data1
del model1
del X_train1 
del Y_train1 
del X_valid1 
del Y_valid1 
del X_test1
del Y_pred1
del Y_test1
gc.collect();
from tensorflow.keras.layers import Input, Embedding, add, Flatten, concatenate, Dropout, Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

data2 = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'month',
    'days'
]]
n_blocks = data2.date_block_num.nunique()
n_shops = data2.shop_id.nunique()
n_items = data2.item_id.nunique()
n_cities = data2.city_code.nunique()
n_months = data2.month.nunique()
n_days = data2.days.nunique()

n_blocks, n_shops, n_items, n_cities , n_months, n_days
#set embedding size to be tthe min between squared root of #unique_values or 20
emb_size = [4, 5, 20, 5, 3, 1]
X_train2 = data2[data2.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train2 = data2[data2.date_block_num < 33]['item_cnt_month']
X_valid2 = data2[data2.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid2 = data2[data2.date_block_num == 33]['item_cnt_month']
X_test2 = data2[data2.date_block_num == 34].drop(['item_cnt_month'], axis=1)

def embedding_input(name, n_in, n_out, reg):
    inp = Input(shape=(1,), dtype='int64', name=name)
    return inp, Embedding(n_in, n_out, input_length=1, embeddings_regularizer=l2(reg))(inp)

blocks_in, b = embedding_input('blocks_in', n_blocks, 4, 1e-4)
shops_in, s = embedding_input('shops_in', n_shops, 10, 1e-4)
items_in, i = embedding_input('items_in', n_items, 20, 1e-4)
cities_in, c = embedding_input('cities_in', n_cities, 5, 1e-4)
months_in, m = embedding_input('months_in',n_months , 3, 1e-4)
days_in, d = embedding_input('days_in', n_days, 1, 1e-4)


blocks_inp = Input(shape=(1,),dtype='int64')
shops_inp = Input(shape=(1,),dtype='int64')
items_inp = Input(shape=(1,),dtype='int64')
cities_inp = Input(shape=(1,),dtype='int64')
months_inp = Input(shape=(1,),dtype='int64')
days_inp = Input(shape=(1,),dtype='int64')

blocks_emb = Embedding(35,4,input_length=1, embeddings_regularizer=l2(1e-6))(blocks_inp)
shops_emb = Embedding(n_shops+10,5,input_length=1, embeddings_regularizer=l2(1e-6))(shops_inp)
items_emb = Embedding(22170 ,20,input_length=1, embeddings_regularizer=l2(1e-6))(items_inp)
cities_emb = Embedding(n_cities+10,5,input_length=1, embeddings_regularizer=l2(1e-6))(cities_inp)
months_emb = Embedding(n_months+10,3,input_length=1, embeddings_regularizer=l2(1e-6))(months_inp)
days_emb = Embedding(n_days+30,1,input_length=1, embeddings_regularizer=l2(1e-6))(days_inp)
import tensorflow.keras.backend as K

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 
x = concatenate([blocks_emb,shops_emb,items_emb,cities_emb,months_emb,days_emb])
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(30, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)
nn = Model([blocks_inp, shops_inp, items_inp, cities_inp, months_inp, days_inp], x)
nn.compile(Adam(0.001), loss=root_mean_squared_error)
nn.summary()

from tensorflow.keras.callbacks import *
def set_callbacks(description='run1',patience=15,tb_base_logdir='./logs/'):
    cp = ModelCheckpoint('best_model_weights_{}.h5'.format(description),save_best_only=True)
    es = EarlyStopping(patience=patience,monitor='val_loss')
    rlop = ReduceLROnPlateau(patience=5)   
    tb = TensorBoard(log_dir='{}{}'.format(tb_base_logdir,description))
    cb = [cp,es,tb,rlop]
    return cb

nn.fit([X_train2.date_block_num, X_train2.shop_id, X_train2.item_id, X_train2.city_code, X_train2.month, X_train2.days], Y_train2, epochs=5, 
          validation_data=([X_valid2.date_block_num, X_valid2.shop_id, X_valid2.item_id, X_valid2.city_code, X_valid2.month, X_valid2.days], Y_valid2), 
          callbacks=set_callbacks())

from tensorflow.keras.models import save_model, load_model

nn.save('nn_model')
# DO NOT RUN - NEED IT AS FEATURE EXTRACTION FOR XGBOOST

del data2
#del X_train2 
#del Y_train2 
#del X_valid2 
#del Y_valid2 
#del X_test2
#del Y_test2
#del nn
gc.collect()
data3 = data[[
    'date_block_num',
    'shop_id',
    'item_id',
    'item_cnt_month',
    'city_code',
    'item_category_id',
    'type_code',
    'subtype_code',
    'item_cnt_month_lag_1',
    'item_cnt_month_lag_2',
    'item_cnt_month_lag_3',
    'item_cnt_month_lag_6',
    'item_cnt_month_lag_12',
    'date_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_1',
    'date_item_avg_item_cnt_lag_2',
    'date_item_avg_item_cnt_lag_3',
    'date_item_avg_item_cnt_lag_6',
    'date_item_avg_item_cnt_lag_12',
    'date_shop_avg_item_cnt_lag_1',
    'date_shop_avg_item_cnt_lag_2',
    'date_shop_avg_item_cnt_lag_3',
    'date_shop_avg_item_cnt_lag_6',
    'date_shop_avg_item_cnt_lag_12',
    'date_cat_avg_item_cnt_lag_1',
    'date_shop_cat_avg_item_cnt_lag_1',
    'date_city_avg_item_cnt_lag_1',
    'date_item_city_avg_item_cnt_lag_1',
    'delta_price_lag',
    'month',
    'days',
    'item_shop_last_sale',
    'item_last_sale',
    'item_shop_first_sale',
    'item_first_sale',
]]
X_train3 = data3[data3.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train3 = data3[data3.date_block_num < 33]['item_cnt_month']
X_valid3 = data3[data3.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid3 = data3[data3.date_block_num == 33]['item_cnt_month']
X_test3 = data3[data3.date_block_num == 34].drop(['item_cnt_month'], axis=1)
emb_feat = ['date_block_num','shop_id','item_id','city_code','month','days'] 
nonemb_feat = list(set(data.columns)-set(emb_feat)-set(['cnt','date_subtype_avg_item_cnt_lag_1','date_shop_subtype_avg_item_cnt_lag_1','date_type_avg_item_cnt_lag_1',
                                                        'date_shop_type_avg_item_cnt_lag_1','delta_revenue_lag_1','item_cnt_month'])) 
x_train_emb = [X_train3[feat] for feat in emb_feat]
x_valid_emb = [X_valid3[feat] for feat in emb_feat]
x_test_emb =  [X_test3[feat] for feat in emb_feat]

x_train_nonemb = [np.reshape(np.array(X_train3[feat]),(len(X_train3[feat]),1,1)) for feat in nonemb_feat]
x_valid_nonemb = [np.reshape(np.array(X_valid3[feat]),(len(X_valid3[feat]),1,1)) for feat in nonemb_feat]
x_test_nonemb = [np.reshape(np.array(X_test3[feat]),(len(X_test3[feat]),1,1)) for feat in nonemb_feat]
nonemb_inps = []
for i in range(len(nonemb_feat)):
      nonemb_inps.extend([Input(shape=(1,1,),dtype='float32')]) 
x = concatenate([blocks_emb,shops_emb,items_emb,cities_emb,months_emb,days_emb]+nonemb_inps)
x = Flatten()(x)
x = Dropout(0.3)(x)
x = Dense(30, activation='relu')(x)
x = Dropout(0.75)(x)
x = Dense(1)(x)
nn2 = Model([blocks_inp, shops_inp, items_inp, cities_inp, months_inp, days_inp]+nonemb_inps, x)
nn2.compile(Adam(0.001), loss=root_mean_squared_error)
nn2.summary()
nn2.fit(x_train_emb+x_train_nonemb, Y_train3, epochs=5,validation_data=(x_valid_emb+x_valid_nonemb, Y_valid3),
          callbacks=set_callbacks())

nn2.save('nn2_model')
# Submit 
y_test3 = nn2.predict(x_test_emb+x_test_nonemb).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": y_test3.flatten()
})
submission.to_csv('cat_emb_and_new_feat_model.csv', index=False)

# save model and predictions (for future, if needed)
pickle.dump(y_test3, open('emb_and_new_feat_test3.pickle', 'wb'))
# DO NOT RUN 

del data3
#del X_train3 
#del Y_train3 
#del X_valid3 
#del Y_valid3 
#del X_test3
#del Y_test3
#del X_train_new_feat
#del X_valid_new_feat
#del X_test_new_feat
gc.collect()
# remove the output layer
nn_model = load_model('nn_model',compile=False)
nn_model._layers.pop()
nn_model.compile(Adam(0.001), loss=root_mean_squared_error)
nn_model.summary()

feature_train1 = nn_model.predict([X_train2.date_block_num, X_train2.shop_id, X_train2.item_id, X_train2.city_code, X_train2.month, X_train2.days])
feature_val1 = nn_model.predict([X_valid2.date_block_num, X_valid2.shop_id, X_valid2.item_id, X_valid2.city_code, X_valid2.month, X_valid2.days])

ts = time.time()

model5 = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model5.fit(
    feature_train1, 
    Y_train2, 
    eval_metric="rmse", 
    eval_set=[(feature_train1, Y_train2), (feature_val1, Y_valid2)], 
    verbose=True, 
    early_stopping_rounds = 5)

time.time() - ts
feature_test1 = nn_model.predict([X_test2.date_block_num, X_test2.shop_id, X_test2.item_id, X_test2.city_code, X_test2.month, X_test2.days])

Y_test5 = model5.predict(feature_test1).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test5
})
submission.to_csv('xgboost_feat_ext1_submission.csv', index=False)

pickle.dump(Y_test5, open('feat_ext_test1.pickle', 'wb'))
from tensorflow.keras.models import load_model

# remove the output layer
nn_model2 = load_model('nn2_model',compile=False)
nn_model2._layers.pop()
nn_model2.compile(Adam(0.001), loss=root_mean_squared_error)
nn_model2.summary()
feature_train2 = nn_model2.predict(x_train_emb+x_train_nonemb)
feature_val2 = nn_model2.predict(x_valid_emb+x_valid_nonemb)
ts = time.time()

model7 = XGBRegressor(
    max_depth=8,
    n_estimators=1000,
    min_child_weight=300, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.3,    
    seed=42)

model7.fit(
    feature_train2, 
    Y_train3, 
    eval_metric="rmse", 
    eval_set=[(feature_train2, Y_train3), (feature_val2, Y_valid3)], 
    verbose=True, 
    early_stopping_rounds = 10)

time.time() - ts
feature_test2 = nn_model2.predict(x_test_emb+x_test_nonemb)
Y_test7 = model7.predict(feature_test2).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test7
})
submission.to_csv('xgboost_feat_ext2_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_test7, open('feat_ext_test2.pickle', 'wb'))
