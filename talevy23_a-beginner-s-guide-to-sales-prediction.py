# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

from itertools import product

import sklearn

import scipy.sparse 

import lightgbm 

import gc



import lightgbm as lgb

from sklearn.linear_model import LinearRegression

from sklearn.metrics import r2_score

from tqdm import tqdm_notebook

from sklearn.metrics import mean_squared_error



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
def downcast_dtypes(df):

    '''

        Changes column types in the dataframe: 

                

                `float64` type to `float32`

                `int64`   type to `int32`

    '''

    

    # Select columns to downcast

    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols =   [c for c in df if df[c].dtype == "int64"]

    

    # Downcast

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols]   = df[int_cols].astype(np.int32)

    

    return df
sales_train = pd.read_csv('../input/sales_train.csv')

print('sales_train')

display(sales_train.head())



test = pd.read_csv('../input/test.csv')

print('test')

display(test.head())



items = pd.read_csv('../input/items.csv')

print('items')

display(items.head())



item_categories = pd.read_csv('../input/item_categories.csv')

print('item_categories')

display(item_categories.head())



shops = pd.read_csv('../input/shops.csv')

print('shops')

display(shops.head())



sample_submission = pd.read_csv('../input/sample_submission.csv')

print('sample_submission')

display(sample_submission.head())
print('sales_train')

display(sales_train.describe(include='all').T)



print('test')

display(test.describe(include='all').T)



print('items')

display(items.describe(include='all').T)



print('item_categories')

display(item_categories.describe(include='all').T)



print('shops')

display(shops.describe(include='all').T)
for col in ['item_price','item_cnt_day']:

    plt.figure()

    plt.title(col)

    sns.boxplot(x=sales_train[col]);

shape0 = sales_train.shape[0] # train size before dropping values

for col in ['item_price','item_cnt_day']:

    max_val = sales_train[col].quantile(.99) # get 99th percentile value

    sales_train = sales_train[sales_train[col]<max_val] # drop outliers

    print(f'{shape0-sales_train.shape[0]} {col} values over {max_val} removed')



print(f'new training set has {sales_train.shape[0]} records')
sales_train[sales_train['item_price']<=0]
sales_train=sales_train[sales_train['item_price']>0]
# Create "grid" with columns

index_cols = ['shop_id', 'item_id', 'date_block_num']



# For every month we create a grid from all shops/items combinations from that month

grid = [] 

for block_num in sales_train['date_block_num'].unique():

    cur_shops = sales_train.loc[sales_train['date_block_num'] == block_num, 'shop_id'].unique()

    cur_items = sales_train.loc[sales_train['date_block_num'] == block_num, 'item_id'].unique()

    grid.append(np.array(list(product(*[cur_shops, cur_items, [block_num]])),dtype='int32'))



# Turn the grid into a dataframe

grid = pd.DataFrame(np.vstack(grid), columns = index_cols,dtype=np.int32)
print(grid.shape)

grid.head()
# latest month

grid['date_block_num'].max()
# append next month

test['date_block_num'] = 34

# add to grid

grid = pd.concat([grid, test[grid.columns]], ignore_index=True, sort=False)

print('grid shape: ',grid.shape)

print('missing values:',grid.isna().sum())

index_cols = ['shop_id', 'item_id', 'date_block_num']



# Groupby data to get shop-item-month aggregates

gb = sales_train.groupby(index_cols,as_index=False).agg({'item_cnt_day':{'target':'sum','trips':'size'}})

# Fix column names

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 

# Join it to the grid

all_data = pd.merge(grid, gb, how='left', on=index_cols).fillna(0)



# Same as above but with shop-month aggregates

gb = sales_train.groupby(['shop_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_shop':'sum','trips_shop':'size'}})

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['shop_id', 'date_block_num']).fillna(0)



# Same as above but with item-month aggregates

gb = sales_train.groupby(['item_id', 'date_block_num'],as_index=False).agg({'item_cnt_day':{'target_item':'sum','trips_item':'size'}})

gb.columns = [col[0] if col[-1] == '' else col[-1] for col in gb.columns.values]

all_data = pd.merge(all_data, gb, how='left', on=['item_id', 'date_block_num']).fillna(0)



all_data.head()
# median item monthly price (using median to avoid outliers)

gb = sales_train.groupby(['date_block_num','item_id'],as_index=False).agg({'item_price':{'median_item_price':'median'}})

# Fix column names

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 

# Join it to the grid

all_data = pd.merge(all_data, gb, how='left', on=['date_block_num','item_id'])



# make sure no na values

print('na median_item_price:',all_data['median_item_price'].isna().sum())

gb = all_data.groupby(['item_id'],as_index=False).agg({'date_block_num':{'item_first_month':'min'}})

# Fix column names

gb.columns = [col[0] if col[-1]=='' else col[-1] for col in gb.columns.values] 

# Join it to the grid

all_data = pd.merge(all_data, gb, how='left', on=['item_id'])
all_data['new_item'] = (all_data['date_block_num']==all_data['item_first_month']).astype(int)

all_data['new_item'].value_counts()
# Downcast dtypes from 64 to 32 bit to save memory

all_data = downcast_dtypes(all_data)

del grid, gb 

gc.collect();

all_data.head()
# List of columns that we will use to create lags

cols_to_rename = list(all_data.columns.difference(index_cols+['item_first_month'])) 



shift_range = [1, 2, 3, 4, 5, 12]



for month_shift in tqdm_notebook(shift_range):

    train_shift = all_data[index_cols + cols_to_rename].copy()

    

    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift

    

    foo = lambda x: '{}_lag_{}'.format(x, month_shift) if x in cols_to_rename else x

    train_shift = train_shift.rename(columns=foo)



    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)



del train_shift



# Don't use old data from year 2013

all_data = all_data[all_data['date_block_num'] >= 12] 



# List of all lagged features

fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]] 

# We will drop these at fitting stage

to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num'] 



# Category for each item

item_category_mapping = items[['item_id','item_category_id']].drop_duplicates()



all_data = pd.merge(all_data, item_category_mapping, how='left', on='item_id')

all_data = downcast_dtypes(all_data)

gc.collect();
all_data.head(5)
item_target_enc_na = .3343 # default na replacement

# Expanding Mean

cumsum = all_data.groupby('item_id')['target'].cumsum() - all_data['target']

cumcnt = all_data.groupby('item_id')['target'].cumcount()



all_data['item_target_enc'] = cumsum/cumcnt

all_data['item_target_enc'].fillna(item_target_enc_na,inplace=True)

corr = np.corrcoef(all_data['target'].values, all_data['item_target_enc'])[0][1]

print(corr)
item_target_enc_na = .3343 # default na replacement

# Expanding Mean

cumsum = all_data.groupby('shop_id')['target'].cumsum() - all_data['target']

cumcnt = all_data.groupby('shop_id')['target'].cumcount()



all_data['shop_id_enc'] = cumsum/cumcnt

all_data['shop_id_enc'].fillna(item_target_enc_na,inplace=True)

corr = np.corrcoef(all_data['target'].values, all_data['shop_id_enc'])[0][1]

print(corr)
all_data.drop(columns='shop_id_enc',inplace=True)

all_data.columns
# figure out difference between month and month

sales_train[['date','date_block_num']].sample(5)
# rule seems to be date_block_num%12+1

sales_train['month'] = sales_train['date_block_num']%12+1

sales_train[['date','month','date_block_num']].sample(5)

# add this to all_data

all_data['month'] = all_data['date_block_num']%12+1
items.groupby('item_category_id',as_index=False)['item_id'].count().rename(columns={'item_id':'total_items'}).describe()
all_data['item_category_id'] = all_data['item_id'].map(items.set_index('item_id')['item_category_id'])
all_data['item_category_id'].isna().sum() # no missing categories
item_target_enc_na = 0 # default na replacement

# Expanding Mean

cumsum = all_data.groupby('item_category_id')['target'].cumsum() - all_data['target']

cumcnt = all_data.groupby('item_category_id')['target'].cumcount()



item_category_id_enc = cumsum/cumcnt

item_category_id_enc.fillna(item_target_enc_na,inplace=True)

all_data['item_category_id_enc'] = item_category_id_enc

corr = np.corrcoef(all_data['target'].values, item_category_id_enc)[0][1]

print(corr)
words = ' '.join(shops['shop_name']).split(' ')

from collections import Counter

c = Counter(words)

c.most_common(10)
shop_by_store = sales_train.groupby('shop_id',as_index=False)['item_cnt_day'].sum()

shop_by_store = shop_by_store.merge(shops, on='shop_id')

print(shop_by_store['shop_name'].isna().sum())

shop_by_store.head()
shop_by_store['name_array'] = shop_by_store['shop_name'].str.split(' ')

top_words = [x for x,y in c.most_common(6)] # common words in shop name

for w in top_words:

    shop_by_store[w] = shop_by_store['shop_name'].map(lambda x: 1 if w in x else 0)

#     shop_by_store[w] = w in shop_by_store['shop_name'].str.split(' ')
for w in top_words:

    print(shop_by_store.groupby(by=w)['item_cnt_day'].mean())
top_words = top_words[0:4] # important shop features

shops['name_array'] = shops['shop_name'].str.split(' ')

for w in top_words:

    shops[w] = shops['shop_name'].map(lambda x: 1 if w in x else 0)



all_data = pd.merge(all_data,shops[['shop_id']+top_words],on='shop_id',how='left') # merge



print(all_data[top_words].isna().sum()) # make sure no nulls

all_data.head()
leaking_columns = ['median_item_price','date_block_num','target','target_shop','target_item','trips','trips_shop','trips_item']



X_train = all_data.loc[all_data['date_block_num'] < 33].drop(leaking_columns, axis=1)

X_val = all_data.loc[all_data['date_block_num'] == 33].drop(leaking_columns, axis=1)

X_test = all_data.loc[all_data['date_block_num'] == 34].drop(leaking_columns, axis=1)



y_train = all_data.loc[all_data['date_block_num'] < 33,'target'].values

y_val = all_data.loc[all_data['date_block_num'] == 33,'target'].values



# save all_data

# all_data.to_csv('all_data.csv',index=False)

del all_data

gc.collect();



X_train.tail()
def validation_function(y_pred,y_true):

    print(f'rmse before [0,20] clipping: {mean_squared_error(y_true, y_pred)}')

    y_pred = y_pred.clip(0,20)

    y_true = y_true.clip(0,20)

    print(f'rmse after [0,20] clipping: {mean_squared_error(y_true, y_pred)}')

    return 

    
# linear regression



lr = LinearRegression()

lr.fit(X_train.values, y_train)

# (due to memory issues we train on half the data)

# lr.fit(X_train[round(X_train.shape[0]/2):-1].values, y_train[round(X_train.shape[0]/2):-1])

pred_lr = lr.predict(X_val.values)



print('Test R-squared for linreg is %f' % r2_score(y_val, pred_lr))

validation_function(y_val,pred_lr)
# create dataset for lightgbm

lgb_train = lgb.Dataset(X_train, y_train)

lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)



# specify your initial configurations as a dict

params = {

    'boosting_type': 'gbdt',

    'objective': 'regression',

    'metric': {'root_mean_squared_error'},

    'num_leaves': 31,

    'learning_rate': 0.05,

    'feature_fraction': 0.9,

    'bagging_fraction': 0.8,

    'bagging_freq': 5,

    'verbose': 10

}

gbm = lgb.train(params,

                lgb_train,

                num_boost_round=15,

                valid_sets=[lgb_train,lgb_eval],

                valid_names=['train','val'],

                early_stopping_rounds=5)



print(gbm.pandas_categorical)

lgb.plot_importance(gbm,figsize=(10,10));
categorical_features = ['shop_id','item_id','item_category_id','new_item']+top_words



lgb_train = lgb.Dataset(X_train, y_train,categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,categorical_feature=categorical_features)



gbm2 = lgb.train(params,

                lgb_train,

                num_boost_round=15,

                valid_sets=[lgb_train,lgb_eval],

                valid_names=['train','val'],

                categorical_feature = categorical_features,

                early_stopping_rounds=5)



print(gbm2.pandas_categorical)

lgb.plot_importance(gbm2,figsize=(10,10));

lgb_train = lgb.Dataset(X_train, y_train,categorical_feature=categorical_features)

lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train,categorical_feature=categorical_features)



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

               'verbose':10 

              }



model = lgb.train(lgb_params,

                lgb_train,

                num_boost_round=1000,

                valid_sets=[lgb_train,lgb_eval],

                valid_names=['train','val'],

                categorical_feature = categorical_features,

                early_stopping_rounds=5)





print(model.pandas_categorical)

lgb.plot_importance(model,figsize=(10,10));



pred_lgb = model.predict(X_val)



print('Test R-squared for LightGBM is %f' % r2_score(y_val, pred_lgb))

validation_function(y_val,pred_lgb)
# train

X_train_level2 = np.c_[model.predict(X_train), lr.predict(X_train.values)] 

lr2 = LinearRegression()

lr2.fit(X_train_level2, y_train)



# predict

X_val_level2 = np.c_[model.predict(X_val), lr.predict(X_val.values)] 

pred_lr2 = lr2.predict(X_val_level2)



validation_function(y_val,pred_lr2)

# lightgbm and lr predicitons

pred_lgb = model.predict(X_test).clip(0,20)

pred_lr = lr.predict(X_test.values).clip(0,20)



# ensamble predicitons

X_test_level2 = np.c_[pred_lgb, pred_lr] 

pred_ensamble = lr2.predict(X_test_level2)
# make sure results are in the same order as the original test set

(test[['shop_id','item_id']].values == X_test[['shop_id','item_id']].values).all()
pd.DataFrame(data={'ID':test['ID'],'item_cnt_month':pred_lgb}).to_csv('lgbm_predictions.csv',index=False)

pd.DataFrame(data={'ID':test['ID'],'item_cnt_month':pred_ensamble}).to_csv('ensamble_predictions.csv',index=False)
