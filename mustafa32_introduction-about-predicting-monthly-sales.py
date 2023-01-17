import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime

import gc

from itertools import product

import lightgbm as lgb

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import LabelEncoder

import time

from statsmodels.tsa.stattools import acf

data_path = '/kaggle/input/competitive-data-science-predict-future-sales/'
def downcast_dtypes(df):

    start_size = df.memory_usage(deep = True).sum() / 1024**2

    print('Memory usage: {:.2f} MB'.format(start_size))



    float_cols = [c for c in df if df[c].dtype == "float64"]

    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]

    df[float_cols] = df[float_cols].astype(np.float32)

    df[int_cols] = df[int_cols].astype(np.int32)

    end_size = df.memory_usage(deep = True).sum() / 1024**2

    print('New Memory usage: {:.2f} MB'.format(end_size))

    return df



def create_record_for_features(df, attrs, target, time_col, aggfunc = np.sum, fill = 0):

    target_for_attrs = df.pivot_table(index = attrs,

                                   values = target, 

                                   columns = time_col, 

                                   aggfunc = aggfunc, 

                                   fill_value = fill,

                                  ).reset_index()

    target_for_attrs.columns = target_for_attrs.columns.map(str)

    target_for_attrs = target_for_attrs.reset_index(drop = True).rename_axis(None, axis = 1)

    return target_for_attrs



def display_df_info(df, name):

    print('-----------Shape of '+ name + '-------------')

    print(df.shape)

    print('-----------Missing values---------')

    print(df.isnull().sum())

    print('-----------Null values------------')

    print(df.isna().sum())

    print('-----------Data types-------------')

    print(df.dtypes)

    print('-----------Memory usage (MB)------')

    print(np.round(df.memory_usage(deep = True).sum() / 1024**2, 2))
sales = pd.read_csv(data_path + 'sales_train.csv')

sales = downcast_dtypes(sales)
items = pd.read_csv(data_path + 'items.csv')

items = downcast_dtypes(items)
item_categories = pd.read_csv(data_path + 'item_categories.csv')

item_categories = downcast_dtypes(item_categories)
shops = pd.read_csv(data_path + 'shops.csv')

shops = downcast_dtypes(shops)
test = pd.read_csv(data_path + 'test.csv')

test = downcast_dtypes(test)
display_df_info(sales, 'Sales')
display_df_info(items, 'items')
display_df_info(item_categories, 'item Categories')
display_df_info(shops, 'shops')
display_df_info(test, 'Test set')
sales_sampled = sales.sample(n = 10000)

sns.pairplot(sales_sampled[['date_block_num', 'shop_id', 'item_id', 'item_price', 'item_cnt_day']], diag_kind = 'kde')

plt.show()
del sales_sampled

gc.collect()
sns.boxplot(x = sales['item_price'])

plt.show()
sales.loc[:, 'item_price'] = sales.loc[:, 'item_price'].clip(-1, 10**5)

sale_with_negative_price = sales[sales['item_price'] < 0]

sale_with_negative_price
sale = sales[(sales.shop_id == 32) & (sales.item_id == 2973) & (sales.date_block_num == 4) & (sales.item_price > 0)]

median = sale.item_price.median()

sales.loc[sales.item_price < 0, 'item_price'] = median
del sale 

del median

del sale_with_negative_price

gc.collect()
sns.boxplot(sales['item_cnt_day'])

plt.show()
sales_temp = sales[sales['item_cnt_day'] > 500]

print('Sold item outliers')

items[items['item_id'].isin(sales_temp['item_id'].values)].merge(sales_temp[['item_id', 'item_cnt_day', 'date_block_num']], on = 'item_id')
del sales_temp

gc.collect()
print('Number of duplicates:', len(sales[sales.duplicated()]))
sales = sales.drop_duplicates(keep = 'first')

print('Number of duplicates:', len(sales[sales.duplicated()]))
start = time.time()

sales.date = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))

print('First sale took place: ', sales.date.min())

print('Last sale took place: ', sales.date.max())

print('It tooks: ', round(time.time() - start), 'seconds')
start = time.time()

pairs_trans = create_record_for_features(sales, ['shop_id', 'item_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.count_nonzero, fill = 0)

print('It tooks: ', round(time.time() - start), 'seconds')
for month in range(1, 12):

    pairs_temp = pairs_trans[['shop_id', 'item_id']][pairs_trans.loc[:,'0': str(month)].sum(axis = 1) == 0]

    print('From month: 0 until ', month,', ', np.round(100 * len(pairs_temp) / len(pairs_trans), 2), '% of the item/shop pairs have made no sales')
for month in range(21, 33):

    pairs_temp = pairs_trans[['shop_id', 'item_id']][pairs_trans.loc[:,str(month): '33'].sum(axis = 1) == 0]

    print('From month: ', month, ' until month: 33, ', np.round(100 * len(pairs_temp) / len(pairs_trans), 2), '% of the item/shop pairs have made no sales')
del pairs_trans

del pairs_temp

gc.collect()
start = time.time()

items_trans = create_record_for_features(sales, ['item_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.count_nonzero, fill = 0)

print('It tooks: ', round(time.time() - start), 'seconds')
for month in range(1, 12):

    items_temp = items_trans['item_id'][items_trans.loc[:,'0': str(month)].sum(axis = 1) == 0]

    print('From month: 0 until ', month,', ', np.round(100 * len(items_temp) / len(items_trans), 2), '% of the items have made no sales')
for month in range(21, 33):

    items_temp = items_trans['item_id'][items_trans.loc[:,str(month): '33'].sum(axis = 1) == 0]

    print('From month: ', month, ' until month: 33, ', np.round(100 * len(items_temp) / len(items_trans), 2), '% of the items have made no sales')
del items_trans

del items_temp

gc.collect()
start = time.time()

shops_trans = create_record_for_features(sales, ['shop_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.count_nonzero, fill = 0)

print('It tooks: ', round(time.time() - start), 'seconds')
for month in range(1, 12):

    shops_temp = shops_trans['shop_id'][shops_trans.loc[:,'0': str(month)].sum(axis = 1) == 0]

    print('From month: 0 until ', month,', ', np.round(100 * len(shops_temp) / len(shops_trans), 2), '% of the shops have made no sales')
for month in range(21, 33):

    shops_temp = shops_trans['shop_id'][shops_trans.loc[:, str(month): '33'].sum(axis = 1) == 0]

    print('From month: ', month, ' until month: 33, ', np.round(100 * len(shops_temp) / len(shops_trans), 2), '% of the shops have made no sales')
del shops_trans

del shops_temp

gc.collect()

item_id = 20949

shop_ids = [25, 24]

item_sales = sales[(sales['item_id'] == item_id) & (sales['shop_id'].isin(shop_ids))]

fig, axs = plt.subplots(figsize = (10, 6),  constrained_layout=True)

sns.pointplot(x = 'date_block_num', y = 'item_cnt_day', hue = 'shop_id', data = item_sales)

axs.set_title('Sales for item: ' + str(item_id))

plt.show()
print('test shape: ', test.shape)

print('number of items in test set: ', test['item_id'].nunique())

print('number of shops in test set: ', test['shop_id'].nunique())

test['item_id'].nunique() * test['shop_id'].nunique() == len(test)
items_test = set(test.item_id)

items_sales = set(sales.item_id)

item_in_test_and_sales = items_test.intersection(items_sales)

item_in_test_not_sales = set(test.item_id) - items_sales.intersection(items_test)

print('There is sales history for:', np.round(100 * len(item_in_test_and_sales) / len(items_test), 2) , '% items in test set')

print('There is No sales history for:', np.round(100 * len(item_in_test_not_sales) / len(items_test), 2), '% items in test set')
shops_test = set(test.shop_id)

shops_sales = set(sales.shop_id)

shops_in_test_and_sales = shops_test.intersection(shops_sales)

shops_in_test_not_sales = set(test.shop_id) - shops_test.intersection(shops_sales)

print('There is sales history for:', np.round(100 * len(shops_in_test_and_sales) / len(shops_test), 2), '% shops in test set')

print('There is No sales history for:', np.round(100 * len(shops_in_test_not_sales) / len(shops_test), 2), '% shops in test set')
item_shop_test = set(test.item_id.astype(str) + '_' + test.shop_id.astype(str))

item_shop_sales = set(sales.item_id.astype(str) + '_' + sales.shop_id.astype(str))

pairs_with_history = len(item_shop_test.intersection(item_shop_sales) )

pairs_with_no_history = len(shops_in_test_and_sales) * len(item_in_test_not_sales)

just_item_with_history = test.shape[0] - (pairs_with_history + pairs_with_no_history)

print('There is sales history for:', np.round(100 * pairs_with_history / len(test), 2) , '% items in the same shops')

print('There is No sales history for:',  np.round(100 * pairs_with_no_history / len(test), 2), '% shop/item pairs')

print('There is sales history for:',  np.round(100 * just_item_with_history / len(test), 2), '% items but in different shops')
del items_test

del items_sales

del item_in_test_and_sales

del item_in_test_not_sales

del shops_test

del shops_sales

del shops_in_test_and_sales

del shops_in_test_not_sales

del item_shop_test

del item_shop_sales

del pairs_with_history

del pairs_with_no_history

del just_item_with_history

gc.collect()
shops['shop_name'] = shops['shop_name'].apply(lambda x: x.lower()).str.replace('[^\w\s]', '').str.replace('\d+','').str.strip()

shops['shop_city'], shops['shop_name'] = shops['shop_name'].str.split(' ', 1).str

shops['shop_type'] = shops['shop_name'].apply(lambda x: 'мтрц' if 'мтрц' in x else 'трц' if 'трц' in x else 'трк' if 'трк' in x else 'тц' if 'тц' in x else 'тк' if 'тк' in x else 'unkown')

shops.head()
print('Shops number:', shops['shop_id'].nunique())

print('Shop names number:', shops['shop_name'].nunique())

print('Shop cities number:', shops['shop_city'].nunique())

print('Shop types number:', shops['shop_type'].nunique())
sales.loc[sales['shop_id'] == 11, 'shop_id'] = 10

shops.loc[shops['shop_id'] == 11, 'shop_id'] = 10



sales.loc[sales['shop_id'] == 23, 'shop_id'] = 24

shops.loc[shops['shop_id'] == 23, 'shop_id'] = 24



sales.loc[sales['shop_id'] == 0, 'shop_id'] = 57

shops.loc[shops['shop_id'] == 0, 'shop_id'] = 57



sales.loc[sales['shop_id'] == 1, 'shop_id'] = 58

shops.loc[shops['shop_id'] == 1, 'shop_id'] = 58



sales.loc[sales['shop_id'] == 40, 'shop_id'] = 39

shops.loc[shops['shop_id'] == 40, 'shop_id'] = 39



shops = shops.drop_duplicates(subset = 'shop_id')
print('Shops number:', shops['shop_id'].nunique())

print('Shop names number:', shops['shop_name'].nunique())

print('Shop cities number:', shops['shop_city'].nunique())

print('Shop types number:', shops['shop_type'].nunique())
def clean_names(df, cols):

    for col in cols:

        df[col] = df[col].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()

        df[col] = df[col].str.strip()

        df.loc[df[col] == '', col] = 'unknown'
items['item_name'], items['item_type'] = items['item_name'].str.split('[', 1).str

items['item_name'], items['item_subtype'] = items['item_name'].str.split('(', 1).str

clean_names(items, ['item_name', 'item_type', 'item_subtype'])

items = items.fillna('unkown')

items.head()





print('Number of items:', items['item_id'].nunique())

print('Number of item_name:', items['item_name'].nunique())

print('Number of item_type:', items['item_type'].nunique())

print('Number of item_subtype:', items['item_subtype'].nunique())
sales = sales.merge(items[['item_id', 'item_category_id']], on = 'item_id', how = 'left')
time_shift = 14
pairs_sales = create_record_for_features(sales, ['shop_id', 'item_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.sum, fill = 0)
shops_items_sales_temp = pairs_sales.sample(1000)#.nlargest(10**4, columns = [str(itr) for itr in range(27, 33)])

pairs_acf = np.zeros((shops_items_sales_temp.shape[0], time_shift + 1))

for i, (ind, shop_item_sales) in enumerate(shops_items_sales_temp.iterrows()):

    pair = shop_item_sales.loc['0': ]

    acf_12 = acf(pair, nlags = time_shift, fft = True)

    pairs_acf[i, :] = acf_12
avgs = np.mean(pairs_acf, axis = 0)

plt.bar(x = np.arange(time_shift + 1), height = avgs)

plt.title('lag importance of shop/item pairs')

plt.show()
pair_lags = [1, 2, 3, 4, 8, 10, 11, 12]
del shops_items_sales_temp

del pairs_sales

del shop_item_sales

del pair

del acf_12

del pairs_acf

gc.collect()
items_sales = create_record_for_features(sales, ['item_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.sum, fill = 0)
items_acf = np.zeros((items_sales.shape[0], time_shift + 1))

for i, item_sales in items_sales.iterrows():

    item_temp = item_sales.loc['0': ]

    if np.sum(item_temp) != 0:

        acf_12 = acf(item_temp, nlags = time_shift, fft = True)

        items_acf[i, :] = acf_12
avgs = np.mean(items_acf, axis = 0)

plt.bar(x = np.arange(time_shift + 1), height = avgs)

plt.title('lag importance of items')

plt.show()
item_lags = [1, 2, 3, 4, 5, 10, 11, 12]
del items_sales

del item_sales

del items_acf

del item_temp

del acf_12

del avgs

gc.collect()
shops_sales = create_record_for_features(sales, ['shop_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.sum, fill = 0)

shops_acf = np.zeros((shops_sales.shape[0], time_shift + 1))

for i, shop_sales in shops_sales.iterrows():

    shop_temp = shop_sales.loc['0': ]

    acf_12 = acf(shop_temp, nlags = time_shift, fft = True)

    shops_acf[i, :] = acf_12
avgs = np.mean(shops_acf, axis = 0)

plt.bar(x = np.arange(time_shift + 1), height = avgs)

plt.title('lag importance of shops')

plt.show()
shop_lags = [1, 2, 3, 4, 7, 8, 10, 12]
del shops_sales

del shop_sales

del shops_acf

del shop_temp

del acf_12

del avgs

gc.collect()
categories_sales = create_record_for_features(sales, ['item_category_id'], 'item_cnt_day', 'date_block_num', aggfunc = np.sum, fill = 0)

categories_acf = np.zeros((categories_sales.shape[0], time_shift + 1))

for i, category_sales in categories_sales.iterrows():

    category_temp = category_sales.loc['0': ]

    acf_12 = acf(category_temp, nlags = time_shift, fft = True)

    categories_acf[i, :] = acf_12
avgs = np.mean(categories_acf, axis = 0)

plt.bar(x = np.arange(time_shift + 1), height = avgs)

plt.title('lag importance of categories')

plt.show()

category_lags = [1, 2, 3, 4, 5, 6, 12]
del categories_sales

del category_sales

del categories_acf

del category_temp

del acf_12

del avgs

gc.collect()
index_cols = ['shop_id', 'item_id', 'date_block_num'] 



def create_train_set(df, index_cols = index_cols):

    grid = []

    for month in df['date_block_num'].unique():

        curr_shops = df[df['date_block_num'] == month]['shop_id'].unique()

        curr_items = df[df['date_block_num'] == month]['item_id'].unique()

        grid.append(np.array(list(product(*[curr_shops, curr_items, [month]])), dtype = 'int32'))



    grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype = np.int32)



    gb = df.groupby(index_cols, as_index = False).agg({'item_cnt_day':'sum'})

    gb.columns = index_cols + ['target']

    all_data = pd.merge(grid, gb, how = 'left', on = index_cols).fillna(0)

    all_data.sort_values(['date_block_num','shop_id','item_id'], inplace = True)

    all_data.loc[:, 'target'] = all_data['target'].clip(0, 20).astype(np.float32)

    print('Sales data shape:', df.shape)

    print('Generated Train data shape:', all_data.shape)

    del curr_shops

    del curr_items

    del gb

    del grid

    gc.collect()

    return all_data
start = time.time()

grid = create_train_set(sales, index_cols)

test['date_block_num'] = 34

grid = pd.concat([grid, test[['item_id', 'shop_id', 'date_block_num']]], ignore_index = True, sort = False, keys = index_cols)

grid = grid.merge(items[['item_id', 'item_category_id']], on = 'item_id', how = 'left')

print(round(time.time() - start), 'seconds')
def lag_features(df, features, go_back_in_time):

    for month_shift in go_back_in_time:

        df_shift = df[index_cols + features].copy()

        df_shift['date_block_num'] = df_shift['date_block_num'] + month_shift

        lag_cols = lambda x: '{}_lag_{}'.format(x, month_shift) if x in features else x

        df_shift = df_shift.rename(columns = lag_cols)

        df = pd.merge(df, df_shift, on = index_cols, how='left')

    return df



def fast_lag_features(df, features, go_back_in_time): 

    features_sales = create_record_for_features(sales, features, 'item_cnt_day', 'date_block_num', aggfunc = np.sum, fill = 0)

    for month in go_back_in_time:

        max_month = df.date_block_num.max()

        cols = [str(itr) for itr in np.arange(0, max_month)]

        gb = features_sales.melt( id_vars = features, 

                                 var_name = 'date_block_num' , 

                                 value_vars= cols, 

                                 value_name = 'target_' + '_'.join(features) + '_lag_' + str(month)

                                )

        gb.date_block_num = gb.date_block_num.astype(np.int16)

        gb.date_block_num = gb.date_block_num + month

        df = pd.merge(df, gb, on = features + ['date_block_num'], how='left')

    return df
start = time.time()

grid = lag_features(grid, ['target'], pair_lags)

print(round(time.time() - start), 'seconds')
start = time.time()

grid = fast_lag_features(grid, ['item_id'], item_lags)

print(round(time.time() - start), 'seconds')
start = time.time()

grid = fast_lag_features(grid, ['shop_id'], shop_lags)

print(round(time.time() - start), 'seconds')
start = time.time()

grid = fast_lag_features(grid, ['item_category_id'], category_lags)

print(round(time.time() - start), 'seconds')
grid.isnull().sum()
grid = grid[grid['date_block_num'] > 11]

grid = grid.fillna(0)
grid = downcast_dtypes(grid)
dates_train = sales.loc[:, ['date', 'date_block_num']].drop_duplicates()

dates_train = dates_train.reset_index(drop = True)

dates_test = dates_train.loc[dates_train.loc[:, 'date_block_num'] == 34 - 12]

dates_test = dates_test.reset_index(drop=True)

dates_test.loc[:,'date_block_num'] = 34

dates_test.loc[:, 'date'] = dates_test.loc[:, 'date'] + pd.DateOffset(years = 1)

dates_all = pd.concat([dates_train, dates_test])

dates_all.loc[:, 'dow'] = dates_all.loc[:, 'date'].dt.dayofweek

dates_all.loc[:, 'year'] = dates_all.loc[:, 'date'].dt.year

dates_all.loc[:, 'month'] = dates_all.loc[:, 'date'].dt.month



dates_all = pd.get_dummies(dates_all, columns = ['dow'])

dow_col = ['dow_' + str(x) for x in range(7)]

date_features = dates_all.groupby(['year', 'month', 'date_block_num'])[dow_col].agg('sum').reset_index()

date_features.loc[:, 'days_of_month'] = date_features.loc[:, dow_col].sum(axis=1)

date_features.loc[:, 'year'] = date_features.loc[:, 'year'] - 2013

date_features = date_features.loc[:, ['month', 'date_block_num']]
grid = grid.merge(date_features, on = 'date_block_num', how = 'left')
del dates_train

del dates_test

del dates_all

del dow_col

del date_features

gc.collect()
grid['category_shop_inter'] = grid['item_category_id'].astype(str) + '_' + grid['shop_id'].astype(str)

grid.loc[:, 'category_shop_inter'] = LabelEncoder().fit_transform(grid.loc[:, 'category_shop_inter'].values)
grid = downcast_dtypes(grid)
cols_to_drop = ['target', 'date_block_num']

def train_val_test_split(df):

    dates = df['date_block_num']

    last_block = dates.max()

    print('Test `date_block_num` is %d' % last_block)

    print('Validation `date_block_num` is %d' % (last_block - 1))

    print('Train `date_block_num` is < %d' % (last_block - 1))

    print('------------------------------')



    X_train = df.loc[dates < last_block - 1].drop(cols_to_drop, axis = 1)

    X_val = df.loc[dates == last_block - 1].drop(cols_to_drop, axis = 1)

    X_test =  df.loc[dates == last_block].drop(cols_to_drop, axis = 1)



    y_train = df.loc[dates < last_block - 1, 'target'].values

    y_val =  df.loc[dates == last_block - 1, 'target'].values

    

    print('X_train shape: ', X_train.shape)

    print('y_train shape: ', y_train.shape)

    print('------------------------------')

    print('X_val shape: ', X_val.shape)

    print('y_val shape: ', y_val.shape)

    print('------------------------------')

    print('X_test shape: ', X_test.shape)

    print('------------------------------')

    return (X_train, y_train, X_val, y_val, X_test)



def rmse(y, y_hat):

    return np.sqrt(mean_squared_error(y, y_hat))



def create_lgbm_model(X_train, y_train, X_val, y_val, params, cat_feats):

    n_estimators = 8000

    d_train = lgb.Dataset(X_train, y_train)

    d_valid = lgb.Dataset(X_val, y_val)

    watchlist = [d_train, d_valid]

    evals_result = {}

    model = lgb.train(params, 

                      d_train, 

                      n_estimators,

                      valid_sets = watchlist, 

                      evals_result = evals_result, 

                      early_stopping_rounds = 50,

                      verbose_eval = 0,

                      categorical_feature = cat_feats,

                    )

    lgb.plot_metric(evals_result)

    return model



def evaluate_model(model, X_train, y_train, X_val, y_val): 

    y_hat = model.predict(X_train)

    print('Training error;', rmse(y_train, y_hat))

    y_val_hat = model.predict(X_val)

    print('Validation error:', rmse(y_val, y_val_hat))
categorical_features = ['shop_id', 'item_category_id', 'month', 'category_shop_inter']

for col in categorical_features:

    grid.loc[:, col] = grid[col].astype('category')
X_train, y_train, X_val, y_val, X_test = train_val_test_split(grid)
start = time.time()

params = {

  'metric': 'rmse',

  'objective': 'mse',

  'verbose': 0, 

  'learning_rate': 0.1,

  'num_leaves': 31,

  'min_data_in_leaf': 20 ,

  'max_depth': -1,

  'save_binary': True,

  'bagging_fraction': 0.8,

  'bagging_freq': 1,

  'bagging_seed': 2**7, 

  'feature_fraction': 0.8,

}

lgbm_model = create_lgbm_model(X_train, y_train, X_val, y_val, params, categorical_features)

print('it tooks: ', round(time.time() - start), 'seconds')
start = time.time()

evaluate_model(lgbm_model, X_train, y_train, X_val, y_val)

print('it tooks: ', round(time.time() - start), 'seconds')
ax = lgb.plot_importance(lgbm_model, max_num_features = 40, figsize = (8, 10))

plt.show()
y_test_pred = lgbm_model.predict(X_test).clip(0, 20)



submission = pd.DataFrame({

    "ID": test.ID, 

    "item_cnt_month": y_test_pred

})

submission.to_csv('lgbm_submission.csv', index = False)
submission.item_cnt_month.hist()