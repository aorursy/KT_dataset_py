# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from calendar import monthrange
from itertools import product

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import matplotlib.pyplot as plt
import seaborn as sns 
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
submission = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
def downcast_dtypes(df):
    float_cols = [c for c in df if df[c].dtype == "float64"]
    int_cols = [c for c in df if df[c].dtype in ["int64", "int32"]]
    df[float_cols] = df[float_cols].astype(np.float32)
    df[int_cols] = df[int_cols].astype(np.int16)
    return df
items = downcast_dtypes(items)
categories = downcast_dtypes(categories)
shops = downcast_dtypes(shops)
test = downcast_dtypes(test)
train = downcast_dtypes(train)
train.head()
train.shape
test.head()
test.shape
submission.head()
print("Null Values:\n train\n{},\n\n test\n{},\n\n items\n{},\n\n categories\n{},\n\n shops\n{}, \n".format(train.isnull().sum(), test.isnull().sum(), items.isnull().sum(), categories.isnull().sum(), shops.isnull().sum()))
train.dtypes
train["date"] = pd.to_datetime(train.date, format="%d.%m.%Y")
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(columns=['date'])
print(train.info(), shops.info(), items.info(), categories.info())
shops.head()
items.head()
categories.head()
print(train.info(), shops.info(), items.info(), categories.info())
fig = plt.figure(figsize=(18,5))
train['shop_id'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)
plt.title('Shop ID Values in the Training Set (Normalized)')
plt.show()
fig = plt.figure(figsize=(18,7))
plt.subplot2grid((3,3), (1,0))
train['item_id'].plot(kind='hist', color = 'green')
plt.title('Item ID Histogram')

plt.subplot2grid((3,3), (1,1))
train['item_price'].plot(kind='hist', color = 'blue')
plt.title('Item Price Histogram')

plt.subplot2grid((3,3), (1,2))
train['item_cnt_day'].plot(kind='hist', color = 'red')
plt.title('Item Count Day Histogram')

fig = plt.figure(figsize=(18,5))
train['date_block_num'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)
plt.title('Month (date_block_num) Values in the Training Set (Normalized)')
train['item_id'].value_counts(ascending = False).head()
items.loc[items['item_id']==20949]
categories.loc[categories['item_category_id']==71]
test.loc[test['item_id']==20949].head()
train['item_cnt_day'].sort_values(ascending=False).head()
train[train['item_cnt_day'] == 2169]
items[items['item_id'] == 11373]
train[train['item_id'] == 11373].head()
train = train[train['item_cnt_day'] < 2000]
train['item_price'].sort_values(ascending=False).head()
train[train['item_price'] == 307980]
items[items['item_id'] == 6066]
train[train['item_id'] == 6066]
train = train[train['item_price'] < 300000]
train['item_price'].sort_values().head()
train[train['item_price'] == -1]
train[train['item_id'] == 2973].head()
price_correct = train[(train['shop_id'] == 32) & (train['item_id'] == 2973) & (train['date_block_num'] == 4) & (train['item_price'] > 0)].item_price.mean()
train.loc[train['item_price'] < 0, 'item_price'] = price_correct
fig = plt.figure(figsize=(18,5))
test['shop_id'].value_counts(normalize=True).plot(kind='bar')
plt.title('Shop ID Values in the Test Set (Normalized)')
plt.show()
test['item_id'].plot(kind='hist')
plt.title('Item ID Histogram - Test Set')

shops_train = train['shop_id'].nunique()
shops_test = test['shop_id'].nunique()
print('Shops in Training Set: ', shops_train)
print('Shops in Test Set: ', shops_test)
shops_train_list = list(train['shop_id'].unique())
shops_test_list = list(test['shop_id'].unique())
set(shops_test_list).issubset(set(shops_train_list))
shops
for i in range(60):
    print('{} - {}'.format(shops.shop_name[i][:10], shops.shop_id[i]))
train.loc[train['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 0, 'shop_id'] = 57

train.loc[train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58

train.loc[train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11

train.loc[train.shop_id == 40, 'shop_id'] = 39
test.loc[test.shop_id == 40, 'shop_id'] = 39
cities = shops['shop_name'].str.split(' ').map(lambda row: row[0])
cities.unique()
shops['city'] = shops['shop_name'].str.split(' ').map(lambda row: row[0])
shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'
shops.head()
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
le.fit_transform(shops['city'])
shops['city_label'] = le.fit_transform(shops['city'])
shops.head()
#shops.drop(['shop_name', 'city'], axis = 1, inplace = True)
items_train = train['item_id'].nunique()
items_test = test['item_id'].nunique()
print('Items in Training Set: ', items_train)
print('Items in Test Set: ', items_test)
items_train_list = list(train['item_id'].unique())
items_test_list = list(test['item_id'].unique())
set(items_test_list).issubset(set(items_train_list))
len(set(items_test_list).difference(items_train_list))
categories_in_test = items.loc[items['item_id'].isin(sorted(test['item_id'].unique()))].item_category_id.unique()
categories.loc[categories['item_category_id'].isin(categories_in_test)]
shops.shop_name.unique()
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops['category'] = shops['shop_name'].str.split(' ').map(lambda x:x[1]).astype(str)

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

category = ['Орджоникидзе,', 'ТЦ', 'ТРК', 'ТРЦ','ул.', 'Магазин', 'ТК', 'склад']
shops.category = shops.category.apply(lambda x: x if (x in category) else 'etc')
shops.category.unique()
shops.groupby(['category']).sum()
category = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК']
shops.category = shops.category.apply(lambda x: x if (x in category) else 'etc')
print('Category Distribution', shops.groupby(['category']).sum())
shops['shop_city'] = shops.city
shops['shop_category'] = shops.category

shops['shop_city'] = le.fit_transform(shops['shop_city'])
shops['shop_category'] = le.fit_transform(shops['shop_category'])

shops = shops[['shop_id','shop_city', 'shop_category']]
shops.head()
print(len(categories.item_category_name.unique()))
categories.item_category_name.unique()
categories['type_code'] = categories.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
categories.loc[(categories.type_code == 'Игровые') | (categories.type_code == 'Аксессуары') | (categories.type_code == 'PC'), 'category'] = 'Игры'

category = ['Игры', 'Карты', 'Кино', 'Книги','Музыка', 'Подарки', 'Программы', 'Служебные', 'Чистые']

categories['type_code'] = categories.type_code.apply(lambda x: x if (x in category) else 'etc')

print(categories.groupby(['type_code']).sum())
categories['type_code'] = le.fit_transform(categories['type_code'])

categories['split'] = categories.item_category_name.apply(lambda x: x.split('-'))
categories['subtype'] = categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
categories['subtype_code'] = le.fit_transform(categories['subtype'])
categories = categories[['item_category_id','type_code', 'subtype_code']]
import re
from collections import Counter
from operator import itemgetter

items['name_1'], items['name_2'] = items['item_name'].str.split('[', 1).str
items['name_1'], items['name_3'] = items['item_name'].str.split('(', 1).str

items['name_2'] = items['name_2'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items['name_3'] = items['name_3'].str.replace('[^A-Za-z0-9А-Яа-я]+', ' ').str.lower()
items = items.fillna('0')

result_1 = Counter(' '.join(items['name_2'].values.tolist()).split(' ')).items()
result_1 = sorted(result_1, key=itemgetter(1))
result_1 = pd.DataFrame(result_1, columns=['feature', 'count'])
result_1 = result_1[(result_1['feature'].str.len() > 1) & (result_1['count'] > 200)]

result_2 = Counter(' '.join(items['name_3'].values.tolist()).split(" ")).items()
result_2 = sorted(result_2, key=itemgetter(1))
result_2 = pd.DataFrame(result_2, columns=['feature', 'count'])
result_2 = result_2[(result_2['feature'].str.len() > 1) & (result_2['count'] > 200)]

result = pd.concat([result_1, result_2])
result = result.drop_duplicates(subset=['feature']).reset_index(drop=True)

print('Most common aditional features:', result)

def name_correction(x):
    x = x.lower()
    x = x.partition('[')[0]
    x = x.partition('(')[0]
    x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
    x = x.replace('  ', ' ')
    x = x.strip()
    return x

items['item_name'] = items['item_name'].apply(lambda x: name_correction(x))
items.name_2 = items.name_2.apply(lambda x: x[:-1] if x != '0' else '0')

items['type'] = items.name_2.apply(lambda x: x[0:8] if x.split(' ')[0] == 'xbox' else x.split(' ')[0])
items.loc[(items.type == 'x360') | (items.type == 'xbox360'), 'type'] = 'xbox 360'
items.loc[items.type == '', 'type'] = 'mac'
items.type = items.type.apply(lambda x: x.replace(' ',''))
items.loc[(items.type == 'pc') | (items.type == 'pс') | (items.type == 'рс'), 'type'] = 'pc'
items.loc[(items.type == 'рs3'), 'type'] = 'ps3'

group_sum = items.groupby('type').sum()
group_sum.loc[group_sum.item_category_id < 200]
drop = ['5c5', '5c7', '5f4', '6dv', '6jv', '6l6', 'android', 'hm3', 'j72', 'kf6', 'kf7','kg4',
            'ps2', 's3v', 's4v'	,'англ', 'русская', 'только', 'цифро']

items.name_2 = items.type.apply(lambda x: 'etc' if x in drop else x)
items = items.drop(['type'], axis=1)
items.groupby('name_2').sum()
items.head()
items['name_2'] = le.fit_transform(items['name_2'])
items['name_3'] = le.fit_transform(items['name_3'])
items.drop(['item_name', 'name_1'], axis=1, inplace=True)
items.head()
train
train.month.dtype
to_append = test[['shop_id', 'item_id']].copy()

to_append['date_block_num'] = train['date_block_num'].max() + 1
to_append['year'] = 2015
to_append['month'] = 11
to_append['item_cnt_day'] = 0
to_append['item_price'] = 0

train = pd.concat([train, to_append], ignore_index=True, sort=False)
train.head()
period = train[['date_block_num', 'year', 'month']].drop_duplicates().reset_index(drop=True)
period['days'] = period.apply(lambda r: monthrange(r.year, r.month)[1], axis=1)

train = train.drop(columns=['month', 'year'])

period.head()
index_cols = ['date_block_num', 'shop_id', 'item_id']
df_new = [] 
for block_num in train['date_block_num'].unique():
    current_shops = train.loc[train['date_block_num'] == block_num, 'shop_id'].unique()
    current_items = train.loc[train['date_block_num'] == block_num, 'item_id'].unique()
    df_new.append(np.array(list(product(*[[block_num], current_shops, current_items])), dtype='int16'))

df_new = pd.DataFrame(np.vstack(df_new), columns = index_cols, dtype = np.int16)
df_new.head()
data = pd.merge(df_new, shops, on='shop_id')
data = pd.merge(data, items, on='item_id')
data = pd.merge(data, categories, on='item_category_id')
data = pd.merge(data, period, on='date_block_num')

data = data[['date_block_num', 'year', 'month', 'days', 'shop_city', 'shop_category', 'shop_id', 'item_category_id', 'type_code', 'subtype_code', 'item_id']] # 'item_price', 'item_cnt_day'

for c in ['date_block_num', 'month', 'days', 'shop_city', 'shop_category', 'shop_id', 'item_category_id', 'type_code', 'subtype_code']:
    data[c] = data[c].astype(np.int8)
data['item_id'] = data['item_id'].astype(np.int16)
data['year'] = data['year'].astype(np.int16)

del df_new, shops, items, categories, to_append

data.head()
temp = train\
.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)\
.agg({'item_cnt_day' : 'sum', 'item_price' : 'mean'})\
.rename(columns= {'item_cnt_day' : 'item_cnt_month', 'item_price' : 'item_price_month'})

temp['item_cnt_month'] = temp['item_cnt_month'].astype(np.float16)
temp['item_price_month'] = temp['item_price_month'].astype(np.float16)

month_summary = pd.merge(data, temp, how='left', on=['date_block_num', 'shop_id', 'item_id'])\
    .fillna(0.0).sort_values(by=['shop_id', 'item_id', 'date_block_num'])

del data, temp

month_summary.head()
print('Min: {} and Max: {} item_cnt_month values'.format(month_summary['item_cnt_month'].min(), month_summary['item_cnt_month'].max()))
month_summary['item_cnt_month'] = month_summary['item_cnt_month'].clip(0,20)
def agg_by(month_summary, group_cols, new_col, target_col = 'item_cnt_month', agg_func = 'mean'):
    aux = month_summary\
        .groupby(group_cols, as_index=False)\
        .agg({target_col : agg_func})\
        .rename(columns= {target_col : new_col})
    aux[new_col] = aux[new_col].astype(np.float16)

    return pd.merge(month_summary, aux, how='left', on=group_cols)

def lag_feature(df, col, lags=[1,2,3,6,12]):
    tmp = df[['date_block_num','shop_id','item_id', col]]
    for i in lags:
        shifted = tmp.copy()
        cols = ['date_block_num','shop_id','item_id', '{}_lag_{}'.format(col, i)]
        shifted.columns = cols
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left').fillna(value={(cols[-1]) : 0.0})
    return df

def agg_by_and_lag(month_summary, group_cols, new_col, lags=[1,2,3,6,12], target_col = 'item_cnt_month', agg_func = 'mean'):
    tmp = agg_by(month_summary, group_cols, new_col, target_col, agg_func)
    tmp = lag_feature(tmp, new_col, lags)
    return tmp.drop(columns=[new_col])
# date_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num'], 'date_avg_item_cnt', [1])

# date_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_id'], 'date_item_avg_item_cnt', [1,2,3,6,12])

# date_city_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_city'], 'date_city_avg_item_cnt', [1])

# date_shop_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id'], 'date_shop_avg_item_cnt', [1,2,3,6,12])

# date_cat_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_category_id'], 'date_cat_avg_item_cnt', [1])

# date_type_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'type_code'], 'date_type_avg_item_cnt', [1])

# date_subtype_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'subtype_code'], 'date_subtype_avg_item_cnt', [1])

# date_shop_category_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category'], 'date_shop_category_avg_item_cnt', [1])

# date_shop_cat_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'item_category_id'], 'date_shop_cat_avg_item_cnt', [1])

# date_shop_type_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'type_code'], 'date_shop_type_avg_item_cnt', [1])

# date_shop_subtype_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'subtype_code'], 'date_shop_subtype_avg_item_cnt', [1])

# date_shop_category_subtype_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category', 'subtype_code'], 'date_shop_category_subtype_avg_item_cnt', [1])

# date_item_city_avg_item_cnt
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_city', 'item_id'], 'date_item_city_avg_item_cnt', [1])

# date_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num'], 'date_avg_item_price', [1], 'item_price_month')

# date_item_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_id'], 'date_item_avg_item_price', [1,2,3,6,12], 'item_price_month')

# date_city_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_city'], 'date_city_avg_item_price', [1], 'item_price_month')

# date_shop_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id'], 'date_shop_avg_item_price', [1,2,3,6,12], 'item_price_month')

# date_cat_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_category_id'], 'date_cat_avg_item_price', [1], 'item_price_month')

# date_type_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'type_code'], 'date_type_avg_item_price', [1], 'item_price_month')

# date_subtype_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'subtype_code'], 'date_subtype_avg_item_price', [1], 'item_price_month')

# date_shop_category_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category'], 'date_shop_category_avg_item_price', [1], 'item_price_month')

# date_shop_cat_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'item_category_id'], 'date_shop_cat_avg_item_price', [1], 'item_price_month')

# date_shop_type_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'type_code'], 'date_shop_type_avg_item_price', [1], 'item_price_month')

# date_shop_subtype_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'subtype_code'], 'date_shop_subtype_avg_item_price', [1], 'item_price_month')

# date_shop_category_subtype_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category', 'subtype_code'], 'date_shop_category_subtype_avg_item_price', [1], 'item_price_month')

# date_item_city_avg_item_price
month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_city', 'item_id'], 'date_item_city_avg_item_price', [1], 'item_price_month')

month_summary['item_shop_first_sale'] = month_summary['date_block_num'] - month_summary.groupby(['item_id','shop_id'])['date_block_num'].transform('min')
month_summary['item_first_sale'] = month_summary['date_block_num'] - month_summary.groupby('item_id')['date_block_num'].transform('min')
month_summary.to_pickle('month_summary.pkl')
month_summary.info()
month_summary = pd.read_pickle('month_summary.pkl')
def generate_subsample(month_summary, target='item_cnt_month'):
    X_test = month_summary[month_summary['date_block_num'] == 34]
    X_test = X_test.drop(columns=[target])

    X_val = month_summary[month_summary['date_block_num'] == 33]
    y_val = X_val[target]
    X_val = X_val.drop(columns=[target])

    X_train = month_summary[(month_summary['date_block_num'] >= 12) & (month_summary['date_block_num'] < 33)]
    y_train = X_train[target]
    X_train = X_train.drop(columns=[target])

    return X_train, y_train, X_val, y_val, X_test
X_train, y_train, X_val, y_val, X_test = generate_subsample(month_summary.drop(columns=['item_price_month']), 'item_cnt_month')

del month_summary
from xgboost import XGBRegressor

def train_gbmodel(X_train, y_train, X_val, y_val):

    RAND_SEED = 42

    lgb_params = {'num_leaves': 2**8, 'max_depth': 19, 'max_bin': 107, #'n_estimators': 3747,
              'bagging_freq': 1, 'bagging_fraction': 0.7135681370918421, 
              'feature_fraction': 0.49446461478601994, 'min_data_in_leaf': 2**8, # 88
              'learning_rate': 0.015980721586917768, 'num_threads': 2, 
              'min_sum_hessian_in_leaf': 6,
              'random_state' : RAND_SEED,
              'bagging_seed' : RAND_SEED,
              'boost_from_average' : 'true',
              'boost' : 'gbdt',
              'metric' : 'rmse',
              'verbose' : 1}

    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val)

    return lgb.train(lgb_params, lgb_train, 
                      num_boost_round=300,
                      valid_sets=[lgb_train, lgb_val],
                      early_stopping_rounds=20)
# model_old_item = train_gbmodel(pd.concat([X_train, X_val]), pd.concat([y_train, y_val]).clip(0, 20), X_val, y_val.clip(0, 20))
gbm_model = train_gbmodel(X_train, y_train, X_val, y_val)

y_hat = gbm_model.predict(X_val).clip(0, 20)
print(np.sqrt(mean_squared_error(y_val.clip(0, 20), y_hat)))

Y_pred = gbm_model.predict(X_val).clip(0, 20)
y_test = gbm_model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": y_test
})
submission.to_csv('gbm_better_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))

model = XGBRegressor(
    max_depth=10,
    n_estimators=1000,
    min_child_weight=0.5, 
    colsample_bytree=0.8, 
    subsample=0.8, 
    eta=0.1,
    tree_method='gpu_hist',
    seed=42)

model.fit(
    X_train, 
    Y_train, 
    eval_metric="rmse", 
    eval_set=[(X_train, Y_train), (X_valid, Y_valid)], 
    verbose=True, 
    early_stopping_rounds = 20)
Y_pred = gbm_model.predict(X_val).clip(0, 20)
y_test = gbm_model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": y_test
})
submission.to_csv('xgb_better_submission.csv', index=False)

# save predictions for an ensemble
pickle.dump(Y_pred, open('xgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('xgb_test.pickle', 'wb'))