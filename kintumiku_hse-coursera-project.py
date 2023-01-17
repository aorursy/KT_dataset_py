import numpy as np 

import pandas as pd

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import gc

from tqdm import tqdm_notebook

from itertools import product

from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_squared_error,r2_score

from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import RidgeCV

from sklearn.svm import LinearSVR

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,VotingRegressor,StackingRegressor

from sklearn.model_selection import KFold,train_test_split

import lightgbm as lgb

import tensorflow as tf

import keras
def downcast_dtypes(df):

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int32)

    return df
train_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')

train_data.head()
test_data=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')

test_data.head()
print(train_data.shape,test_data.shape)
print(train_data['shop_id'].nunique(),test_data['shop_id'].nunique())

l=train_data['shop_id'].unique()

for i in test_data['shop_id'].unique():

    if i not in l:

        print(i)
print(train_data['item_id'].nunique(),test_data['item_id'].nunique())

l=train_data['item_id'].unique()

c=0

for i in test_data['item_id'].unique():

    if i not in l:

        c+=1

print('test item id not in train',c)

l=test_data['item_id'].unique()

c=0

for i in train_data['item_id'].unique():

    if i not in l:

        c+=1

print('train item id not in test',c)
items=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')

print(items.shape)
items.head()
items['item_id'].nunique()
item_dict={}

for i in range(items.shape[0]):

    itid=items.loc[i,'item_id']

    itcat=items.loc[i,'item_category_id']

    item_dict[itid]=itcat
# Downcasting to 16 bit integer

train_data['item_cnt_day'] = train_data['item_cnt_day'].astype('int16')
test_data.isnull().sum()
train_data.isnull().sum()
train_data['date_block_num'].value_counts()
train_data['item_cnt_day'].value_counts()
sns.boxplot(train_data['item_cnt_day'])
train_data['item_price'].value_counts()
sns.boxplot(train_data['item_price'])
train_data.describe()
fig = plt.figure(figsize=(18,9))

plt.subplots_adjust(hspace=.5)



plt.subplot2grid((3,3), (0,0), colspan = 3)

train_data['shop_id'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Shop ID Values in the Training Set (Normalized)')



plt.subplot2grid((3,3), (1,0))

train_data['item_id'].plot(kind='hist', alpha=0.7)

plt.title('Item ID Histogram')



plt.subplot2grid((3,3), (1,1))

train_data['item_price'].plot(kind='hist', alpha=0.7, color='orange')

plt.title('Item Price Histogram')



plt.subplot2grid((3,3), (1,2))

train_data['item_cnt_day'].plot(kind='hist', alpha=0.7, color='green')

plt.title('Item Count Day Histogram')



plt.subplot2grid((3,3), (2,0), colspan = 3)

train_data['date_block_num'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)

plt.title('Month (date_block_num) Values in the Training Set (Normalized)')



plt.show()
print(np.percentile(train_data['item_cnt_day'],99.99))

print(np.percentile(train_data['item_price'],99.99))
train_data = train_data[train_data['item_cnt_day'] < 100]

train_data = train_data[train_data['item_price'] < 30000]
train_data[train_data['item_price']<0]
median = train_data[(train_data.shop_id==32)&(train_data.item_id==2973)&(train_data.date_block_num==4)&(train_data.item_price>0)].item_price.median()

train_data.loc[train_data.item_price<0, 'item_price'] = median
train_data[train_data['item_cnt_day']<0]
# Якутск Орджоникидзе, 56

train_data.loc[train_data.shop_id == 0, 'shop_id'] = 57

test_data.loc[test_data.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

train_data.loc[train_data.shop_id == 1, 'shop_id'] = 58

test_data.loc[test_data.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

train_data.loc[train_data.shop_id == 10, 'shop_id'] = 11

test_data.loc[test_data.shop_id == 10, 'shop_id'] = 11
item_categories=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')

shops=pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
shops.head()
item_categories.head()
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

shops['city_code'] = LabelEncoder().fit_transform(shops['city'])

shops = shops[['shop_id','city_code']]



item_categories['split'] = item_categories['item_category_name'].str.split('-')

item_categories['type'] = item_categories['split'].map(lambda x: x[0].strip())

item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])

# if subtype is nan then type

item_categories['subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())

item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['subtype'])

item_categories = item_categories[['item_category_id','type_code', 'subtype_code']]



items.drop(['item_name'], axis=1, inplace=True)
import time
ts = time.time()

matrix = []

cols = ['date_block_num','shop_id','item_id']

for i in range(34):

    sales = train_data[train_data.date_block_num==i]

    matrix.append(np.array(list(product([i], sales.shop_id.unique(), sales.item_id.unique())), dtype='int16'))

    

matrix = pd.DataFrame(np.vstack(matrix), columns=cols)

matrix['date_block_num'] = matrix['date_block_num'].astype(np.int8)

matrix['shop_id'] = matrix['shop_id'].astype(np.int8)

matrix['item_id'] = matrix['item_id'].astype(np.int16)

matrix.sort_values(cols,inplace=True)

time.time()-ts
train_data['revenue'] = train_data['item_price'] *  train_data['item_cnt_day']
ts = time.time()

group = train_data.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=cols, how='left')

matrix['item_cnt_month'] = (matrix['item_cnt_month']

                                .fillna(0)

                                .clip(0,20) # NB clip target here

                                .astype(np.float16))

time.time() - ts
test_data['date_block_num'] = 34

test_data['date_block_num'] = test_data['date_block_num'].astype(np.int8)

test_data['shop_id'] = test_data['shop_id'].astype(np.int8)

test_data['item_id'] = test_data['item_id'].astype(np.int16)
ts = time.time()

matrix = pd.concat([matrix, test_data], ignore_index=True, sort=False, keys=cols)

matrix.fillna(0, inplace=True) # 34 month

time.time() - ts
ts = time.time()

matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')

matrix = pd.merge(matrix, items, on=['item_id'], how='left')

matrix = pd.merge(matrix, item_categories, on=['item_category_id'], how='left')

matrix['city_code'] = matrix['city_code'].astype(np.int8)

matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)

matrix['type_code'] = matrix['type_code'].astype(np.int8)

matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)

time.time() - ts
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

group = matrix.groupby(['date_block_num', 'type_code']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_type_avg_item_cnt' ]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', 'type_code'], how='left')

matrix['date_type_avg_item_cnt'] = matrix['date_type_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'date_type_avg_item_cnt')

matrix.drop(['date_type_avg_item_cnt'], axis=1, inplace=True)

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

group = matrix.groupby(['date_block_num', 'subtype_code']).agg({'item_cnt_month': ['mean']})

group.columns = [ 'date_subtype_avg_item_cnt' ]

group.reset_index(inplace=True)



matrix = pd.merge(matrix, group, on=['date_block_num', 'subtype_code'], how='left')

matrix['date_subtype_avg_item_cnt'] = matrix['date_subtype_avg_item_cnt'].astype(np.float16)

matrix = lag_feature(matrix, [1], 'date_subtype_avg_item_cnt')

matrix.drop(['date_subtype_avg_item_cnt'], axis=1, inplace=True)

time.time() - ts
matrix['month'] = matrix['date_block_num'] % 12
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
matrix.to_pickle('data.pkl')

del matrix

del group

del items

del shops

del item_categories

del train_data

gc.collect();
data = pd.read_pickle('data.pkl')
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)

y_train = data[data.date_block_num < 33]['item_cnt_month']

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)

y_valid = data[data.date_block_num == 33]['item_cnt_month']

X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
del data

gc.collect();
lr = LinearRegression()

lr.fit(X_train.values, y_train)

pred_lr = lr.predict(X_valid.values).clip(0,20)

test_lr = lr.predict(X_test.values).clip(0,20)

print('Test R-squared for linreg is %f' % r2_score(y_valid, pred_lr))

print('Test rmse for linreg is %f' % mean_squared_error(y_valid, pred_lr,squared=False))

np.save('lr.npy',pred_lr)

np.save('lr_test.npy',test_lr)
lgb_params = {

               'feature_fraction': 0.75,

               'metric': 'rmse',

               'nthread':1, 

               'min_data_in_leaf': 2**7, 

               'bagging_fraction': 0.75, 

               'learning_rate': 0.03, 

               'objective': 'mse', 

               'bagging_seed': 2**7, 

               'num_leaves': 2**7,

               'bagging_freq':1,

               'verbose':0 

              }



model = lgb.train(lgb_params, lgb.Dataset(X_train, label=y_train), 100)

pred_lgb = model.predict(X_valid).clip(0,20)

test_lgb = model.predict(X_test).clip(0,20)

print('Test R-squared for LightGBM is %f' % r2_score(y_valid, pred_lgb))

print('Test rmse for LightGBM is %f' % mean_squared_error(y_valid, pred_lgb,squared=False))

np.save('lgb.npy',pred_lgb)

np.save('lgb_test.npy',test_lgb)
svc = LinearSVR(verbose=1)

svc.fit(X_train, y_train)

pred_svc = svc.predict(X_valid).clip(0,20)

test_svc = svc.predict(X_test).clip(0,20)

print('Test R-squared for xgboost is %f' % r2_score(y_valid, pred_svc))

print('Test rmse for xgboost is %f' % mean_squared_error(y_valid, pred_svc,squared=False))

np.save('svc.npy',pred_svc)

np.save('svc_test.npy',test_svc)
ridge = RidgeCV()

ridge.fit(X_train, y_train)

pred_ridge = ridge.predict(X_valid).clip(0,20)

test_ridge = ridge.predict(X_test).clip(0,20)

print('Test R-squared for xgboost is %f' % r2_score(y_valid, pred_ridge))

print('Test rmse for xgboost is %f' % mean_squared_error(y_valid, pred_ridge,squared=False))

np.save('ridge.npy',pred_ridge)

np.save('ridge_test.npy',test_ridge)
xgb = XGBRegressor(max_depth=8,n_estimators=20,min_child_weight=300, colsample_bytree=0.8, subsample=0.8, eta=0.3)

xgb.fit(X_train, y_train, eval_metric="rmse", eval_set=[(X_train, y_train), (X_valid, y_valid)], verbose=True)

pred_xgb = xgb.predict(X_valid).clip(0,20)

test_xgb = xgb.predict(X_test).clip(0,20)

print('Test R-squared for xgboost is %f' % r2_score(y_valid, pred_xgb))

print('Test rmse for xgboost is %f' % mean_squared_error(y_valid, pred_xgb,squared=False))

np.save('xgb.npy',pred_xgb)

np.save('xgb_test.npy',test_xgb)
grad = GradientBoostingRegressor(criterion='mse',n_estimators=20,subsample=0.8,min_samples_split=100,max_depth=8,verbose=2)

grad.fit(X_train, y_train)

pred_grad = grad.predict(X_valid).clip(0,20)

test_grad = grad.predict(X_test).clip(0,20)

print('Test R-squared for gardient boost is %f' % r2_score(y_valid, pred_grad))

print('Test rmse for gradient boost is %f' % mean_squared_error(y_valid, pred_grad,squared=False))

np.save('grad.npy',pred_grad)

np.save('grad_test.npy',test_grad)
rand = RandomForestRegressor(n_estimators=50,min_samples_split=20,max_depth=8,verbose=2)

rand.fit(X_train, y_train)

pred_rand = rand.predict(X_valid).clip(0,20)

test_rand = rand.predict(X_test).clip(0,20)

print('Test R-squared for random forest is %f' % r2_score(y_valid, pred_rand))

print('Test rmse for random forest is %f' % mean_squared_error(y_valid, pred_rand,squared=False))

np.save('rand.npy',pred_rand)

np.save('rand_test.npy',test_rand)
import tensorflow as tf

from keras.layers import Dense

from keras.models import Sequential

from keras.optimizers import Adam
nn_model=Sequential()

nn_model.add(Dense(1024,activation='relu'))

nn_model.add(Dense(1024,activation='relu'))

nn_model.add(Dense(256,activation='relu'))

nn_model.add(Dense(256,activation='relu'))

nn_model.add(Dense(1,activation='relu'))

nn_model.compile(optimizer=Adam(lr=0.0000001),loss='mean_squared_error',metrics=['mean_squared_error'])
nn_model.fit(X_train,y_train,epochs=2,batch_size=512,validation_data=(X_valid,y_valid))

pred_nn=nn_model.predict(X_valid).clip(0,20)

test_nn=nn_model.predict(X_test).clip(0,20)

print('Test R-squared for neural network is %f' % r2_score(y_valid, pred_nn))

print('Test rmse for neural network is %f' % mean_squared_error(y_valid, pred_nn,squared=False))

np.save('nn.npy',pred_nn)

np.save('nn_test.npy',test_nn)
pred_lr=np.load('/kaggle/input/predictions/lr.npy')

pred_lgb=np.load('/kaggle/input/predictions/lgb.npy')

pred_svc=np.load('/kaggle/input/predictions/svc.npy')

pred_ridge=np.load('/kaggle/input/predictions/ridge.npy')

pred_xgb=np.load('/kaggle/input/predictions/xgb.npy')

pred_grad=np.load('/kaggle/input/predictions/grad.npy')

pred_rand=np.load('/kaggle/input/predictions/rand.npy')

pred_nn=np.load('/kaggle/input/predictions/nn.npy')
test_lr=np.load('/kaggle/input/predictions/lr_test.npy')

test_lgb=np.load('/kaggle/input/predictions/lgb_test.npy')

test_svc=np.load('/kaggle/input/predictions/svc_test.npy')

test_ridge=np.load('/kaggle/input/predictions/ridge_test.npy')

test_xgb=np.load('/kaggle/input/predictions/xgb_test.npy')

test_grad=np.load('/kaggle/input/predictions/grad_test.npy')

test_rand=np.load('/kaggle/input/predictions/rand_test.npy')

test_nn=np.load('/kaggle/input/predictions/nn_test.npy')
from sklearn.model_selection import KFold

data_level2=np.c_[pred_lr,pred_lgb,pred_svc,pred_ridge,pred_xgb,pred_grad,pred_rand,pred_nn]

test_level2=np.c_[test_lr,test_lgb,test_svc,test_ridge,test_xgb,test_grad,test_rand,test_nn]

y_valid=np.array(y_valid)

print(data_level2.shape,y_valid.shape)

print(test_level2.shape,X_test.shape)
kf = KFold(n_splits=5)

i=0

test=np.zeros(test_lgb.shape)

for train_index, val_index in kf.split(data_level2,y_valid):

    X_train, X_val = data_level2[train_index], data_level2[val_index]

    y_train, y_val = y_valid[train_index], y_valid[val_index]

    lr = XGBRegressor(max_depth=5,n_estimators=50,min_child_weight=50, colsample_bytree=0.8, subsample=0.8, eta=0.3)

    lr.fit(X_train, y_train,eval_metric="rmse",eval_set=[(X_train, y_train), (X_val, y_val)], verbose=True)

    val = lr.predict(X_val).clip(0,20)

    test_lr = lr.predict(test_level2).clip(0,20)

    i+=1

    print('Iteration = %f' % i)

    print('Test R-squared for linreg is %f' % r2_score(y_val, val))

    print('Test rmse for linreg is %f' % mean_squared_error(y_val, val,squared=False))

test=test/5

np.save('test.npy',test)
submission = pd.DataFrame({

    "ID": test_data.index, 

    "item_cnt_month": test

})

submission.to_csv('submission.csv', index=False)