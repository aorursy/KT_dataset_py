import pandas as pd

import numpy as np



from itertools import product

import gc



import matplotlib.pyplot as plt

import seaborn as sns

from multiprocessing import Pool



import lightgbm as lgb
from matplotlib import style

style.use('seaborn')



pd.set_option('display.max_columns', 100)

pd.set_option('display.max_rows', 100)
train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')

items = pd.read_csv('../input/competitive-data-science-predict-future-sales/items.csv')

shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv')

categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')
print(train.shape)

train.head(10)
train.drop_duplicates(inplace=True, ignore_index=True)
sns.boxplot(x=train['item_cnt_day'])

plt.show()
sns.boxplot(x=train['item_price'])

plt.show()
train.loc[train['item_cnt_day'].argmax()]
items[items['item_id'] == 11373]
sns.distplot(train[train['item_id']==11373]['item_cnt_day'].values)

plt.show()
train[train['item_cnt_day'] == 1000]
items[items['item_id'] == 20949]
sns.distplot(train[train['item_id']==20949]['item_cnt_day'].values)

plt.show()
train.iloc[train['item_price'].argmax()]
items[items['item_id'] == 6066]
test[test['item_id'] == 6066]
train.iloc[train['item_price'].argmin()]
train.loc[train['item_price'].argmin(), 'item_price'] = train[train['item_id'] == 2973].item_price.mean()
train = train[train['item_cnt_day'] <= 1000]
train = train[train['item_price'] < 300000]
cols = ['date_block_num', 'shop_id', 'item_id']
shops
train.loc[train['shop_id'] == 0, 'shop_id'] = 57

test.loc[test['shop_id'] == 0, 'shop_id'] = 57

train.loc[train['shop_id'] == 1, 'shop_id'] = 58

test.loc[test['shop_id'] == 1, 'shop_id'] = 58

train.loc[train['shop_id'] == 10, 'shop_id'] = 11

test.loc[test['shop_id'] == 10, 'shop_id'] = 11

train.loc[train['shop_id'] == 40, 'shop_id'] = 39

test.loc[test['shop_id'] == 40, 'shop_id'] = 39
%%time

data = []

for block in range(34):

    tmp = train[train['date_block_num'] == block]

    data.append(np.array(list(product([block], tmp['shop_id'].unique(), tmp['item_id'].unique())), dtype='int16'))



del tmp



data = pd.DataFrame(data=np.vstack(data), columns=cols)
data['date_block_num'] = data['date_block_num'].astype('int8')

data['shop_id'] = data['shop_id'].astype('int8')

data.dtypes
data.sort_values(cols, inplace=True)
group = train.groupby(cols).agg({'item_cnt_day': 'sum'})

group.columns = ['target']

group.reset_index(inplace=True)



data = data.merge(group, how='left', on=cols)



data['target'] = data['target'].fillna(0).clip(0, 20).astype('float16')
data.to_hdf('data.hdf5', 'df')
data = pd.read_hdf('data.hdf5', 'df')

data
shops
shops['city_name'] = shops['shop_name'].apply(lambda x: x.split(' ')[0])

shops.replace('!Якутск', 'Якутск', inplace=True)

shops['city_name'], _ = pd.factorize(shops['city_name'])

shops['city_name'] = shops['city_name'].astype('int8')



shops.head(10)
categories['category_general_name'] = categories['item_category_name'].apply(lambda x: x.split(' ')[0])

categories['category_general_name'], _ = pd.factorize(categories['category_general_name'])

categories['category_general_name'] = categories['category_general_name'].astype('int8')



categories.head()
test['date_block_num'] = 34

test['shop_id'] = test['shop_id'].astype('int8')

test['date_block_num'] = test['date_block_num'].astype('int8')

test['item_id'] = test['item_id'].astype('int16')
data = pd.concat([data, test], ignore_index=True).fillna(-1)

data
data = data.merge(shops[['shop_id', 'city_name']], how='left', on='shop_id')
a = pd.merge(items[['item_id', 'item_category_id']], categories[['item_category_id','category_general_name']], how='left', on='item_category_id')

data = data.merge(a, how='left', on='item_id')

del a
data['item_category_id'] = data['item_category_id'].astype('int8')
data['month'] = data['date_block_num'] % 12 + 1
def lag_generator(df, col, lags):

    tmp = df[['date_block_num', 'shop_id', 'item_id', col]]

    for lag in lags:

        a = tmp.copy()

        a['date_block_num'] += lag

        a.columns = ['date_block_num', 'shop_id', 'item_id', f'{col}_lag_{lag}']

        df = df.merge(a, how='left', on=['date_block_num', 'shop_id', 'item_id'])

    return df
%%time

lags = [1, 2, 3]

data = lag_generator(data, 'target', lags)



data.fillna(-1, inplace=True)
%%time

group = data.groupby(['date_block_num', 'shop_id']).agg({'target': 'mean'})

group.columns = ['target_mean_date_shop']

group.reset_index(inplace=True)



data = data.merge(group, how='left', on=['date_block_num', 'shop_id']).fillna(-1)



data = lag_generator(data, 'target_mean_date_shop', [1])

data.drop(columns='target_mean_date_shop', inplace=True)



data.fillna(-1, inplace=True)
%%time

group = data.groupby(['date_block_num', 'item_id']).agg({'target': 'mean'})

group.columns = ['target_mean_date_item']

group.reset_index(inplace=True)



data = data.merge(group, how='left', on=['date_block_num', 'item_id']).fillna(-1)



data = lag_generator(data, 'target_mean_date_item', [1, 2, 3])

data.drop(columns='target_mean_date_item', inplace=True)



data.fillna(-1, inplace=True)
%%time

group = data.groupby(['date_block_num', 'item_category_id']).agg({'target': 'mean'})

group.columns = ['target_mean_date_category']

group.reset_index(inplace=True)



data = data.merge(group, how='left', on=['date_block_num', 'item_category_id']).fillna(-1)



data = lag_generator(data, 'target_mean_date_category', [1])

data.drop(columns='target_mean_date_category', inplace=True)



data.fillna(-1, inplace=True)
%%time

group = data.groupby(['date_block_num', 'city_name']).agg({'target': 'mean'})

group.columns = ['target_mean_date_city']

group.reset_index(inplace=True)



data = data.merge(group, how='left', on=['date_block_num', 'city_name']).fillna(-1)



data = lag_generator(data, 'target_mean_date_city', [1])

data.drop(columns='target_mean_date_city', inplace=True)



data.fillna(-1, inplace=True)
%%time

group = data.groupby(['date_block_num', 'category_general_name']).agg({'target': 'mean'})

group.columns = ['target_mean_date_gencategory']

group.reset_index(inplace=True)



data = data.merge(group, how='left', on=['date_block_num', 'category_general_name']).fillna(-1)



data = lag_generator(data, 'target_mean_date_gencategory', [1])

data.drop(columns='target_mean_date_gencategory', inplace=True)



data.fillna(-1, inplace=True)
data.to_hdf('data1.hdf5', 'df')
data = pd.read_hdf('data1.hdf5', 'df')

data
group = train.groupby('item_id').agg({'item_price': 'mean'})

group['item_price'] = group['item_price'].astype('float32')

group.columns = ['item_mean_price']

group.reset_index(inplace=True)



data = data.merge(group, how='left', on='item_id')
%%time

group = train.groupby(['shop_id', 'item_id'], sort=False)['date_block_num'].unique()



group.name = 'last_sales'
data = data.merge(group.reset_index(), how='left', on=['shop_id', 'item_id'])
def find_prev_sel(arr):

    try:

        date_block = arr[0]

        last_sale = arr[1]

        return last_sale[last_sale < date_block].max()

    except:

        return np.nan
%%time

pool = Pool(2)



data['last_sale'] = pool.map(find_prev_sel, data[['date_block_num', 'last_sales']].values)



pool.close()

pool.join()
group = train.groupby(['item_id', 'date_block_num'], as_index=False, sort=False).agg({'item_price': 'mean'})

group.columns = ['item_id', 'last_sale', 'item_date_mean_price_prev_sale']



data = data.merge(group, how='left', on=['item_id', 'last_sale'])
data['delta_item_prev_price'] = data['item_mean_price'] - data['item_date_mean_price_prev_sale']

data['prev_sold_delta'] = data['date_block_num'] - data['last_sale']
data.drop(columns=['last_sale', 'item_mean_price', 'item_date_mean_price_prev_sale', 'last_sales'], inplace=True)
data.fillna(-1, inplace=True)
data['prev_sold_delta'] = data['prev_sold_delta'].astype('int8')

data['delta_item_prev_price'] = data['delta_item_prev_price'].astype('float32')
train['revenue'] = train['item_price'] * train['item_cnt_day']

group = train.groupby(['date_block_num', 'shop_id']).agg({'revenue': 'sum'})

group.columns = ['revenue_lag_1']

group.reset_index(inplace=True)

group['date_block_num'] += 1



data = data.merge(group, how='left', on=['date_block_num', 'shop_id'])



data.fillna(-1, inplace=True)
data.to_hdf('data2.hdf5', 'df')
data = pd.read_hdf('data2.hdf5', 'df')

data
del train

del items

del test

del shops

del categories

del group



gc.collect()
data = data[data['date_block_num'] >= 3]
train_data = lgb.Dataset(data[data['date_block_num'] < 33].drop(columns=['date_block_num', 'target']), label=data[data['date_block_num'] < 33].target.values, categorical_feature=['shop_id', 'item_id', 'city_name', 'item_category_id', 'category_general_name', 'month', 'prev_sold_delta'])

val_data = lgb.Dataset(data[data['date_block_num'] == 33].drop(columns=['date_block_num', 'target']), label=data[data['date_block_num'] == 33].target.values, categorical_feature=['shop_id', 'item_id', 'city_name', 'item_category_id', 'category_general_name', 'month', 'prev_sold_delta'], reference=train_data)

test_data = data[data['date_block_num'] == 34].drop(columns=['date_block_num', 'target'])
%%time

params = {'metric': 'rmse',

          'learning_rate': 0.01,

          'max_depth': 13,

          'num_leaves': 1673,

          'random_state': 42,

          'num_iterations': 500,

          'early_stopping_round': 12,

          'num_threads': 2

         }



model = lgb.train(params, train_data, valid_sets=[val_data, train_data])



#model = lgb.train(params, train_data, valid_sets=train_data)
lgb.plot_importance(model)
preds = model.predict(test_data)
submission = pd.read_csv('../input/competitive-data-science-predict-future-sales/sample_submission.csv')
submission['item_cnt_month'] = preds

submission
submission.to_csv('best_lgb.csv', index=False)