# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from xgboost import XGBRegressor
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Importing the data 
item = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
shop = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
sales_train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
item_categories = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
testd = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
sampl = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sample_submission.csv')
sampl.head()
# seeing teh basic structure of the dta in teh frame 
data = [item,shop,sales_train,testd,item_categories,sampl]
for i in data:
    print(i.info())
    print('\n')
#Finding Any Null values
sales_train.isna().sum()
# Some Anylysis Seeing Tools Imported
import seaborn as sns
from matplotlib import pyplot as plt
%matplotlib inline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix
#Distribution Of sales Vs Shop Analysis in details and seeing how shop perform
sns.set(rc={'figure.figsize':(40, 40)})
sns.set_context("talk", font_scale=1)
sales_month_shop_id = pd.DataFrame(sales_train.groupby(['shop_id']).sum().item_cnt_day).reset_index()
sales_month_shop_id.columns = ['shop_id', 'sum_sales']
sns.barplot(x ='shop_id', y='sum_sales', data=sales_month_shop_id, palette='Paired')
plt.title('Distribution of sales per shop');
del sales_month_shop_id
#Seeing Items with sales analysis
sales_item_id = pd.DataFrame(sales_train.groupby(['item_id']).sum().item_cnt_day)
plt.xlabel('item id')
plt.ylabel('sales')
plt.plot(sales_item_id)
#Seeing the real max item and its name and its other info
anom_item = sales_item_id.item_cnt_day.argmax()
print(anom_item)
item[item['item_id'] == 20602]
# we will try to plot how does the items matches
sns.set(style = "dark")
plt.plot(sales_train['item_id'], sales_train['item_price'], '*', color='Green');
sales_train[sales_train['item_price'] > 50000]
print(item[item['item_id'] == 6066])
print(item[item['item_id'] == 11365])
print(item[item['item_id'] == 13199])
print(item_categories[item_categories['item_category_id'] == 75])
print(item_categories[item_categories['item_category_id'] == 9])
print(item_categories[item_categories['item_category_id'] == 69])
print(shop[shop['shop_id'] == 12])
print(shop[shop['shop_id'] == 25])
sales_train_sub = sales_train
sales_train_sub['month'] = pd.DatetimeIndex(sales_train_sub['date']).month
sales_train_sub['year'] = pd.DatetimeIndex(sales_train_sub['date']).year
sales_train_sub.head(10)
monthly_sales=sales_train_sub.groupby(["date_block_num","shop_id","item_id"])["item_cnt_day"].agg(item_cnt_day = 'sum')

monthly_sales['date_block_num'] = monthly_sales.index.get_level_values('date_block_num') 
monthly_sales['shop_id'] = monthly_sales.index.get_level_values('shop_id') 
monthly_sales['item_id'] = monthly_sales.index.get_level_values('item_id') 
monthly_sales.reset_index(drop=True, inplace=True)

monthly_sales = monthly_sales.reindex(['date_block_num','shop_id','item_id','item_cnt_day'], axis=1)
monthly_sales.head(10)
fig = plt.figure(figsize=(18,8))
plt.subplots_adjust(hspace=.5)

plt.subplot2grid((3,3), (0,0), colspan = 3)
testd['shop_id'].value_counts(normalize=True).plot(kind='bar', alpha=0.7)
plt.title('Shop ID Values in the Test Set (Normalized)')

plt.subplot2grid((3,3), (1,0))
testd['item_id'].plot(kind='hist', alpha=0.7)
plt.title('Item ID Histogram - Test Set')

plt.show()
# Remove outliers
sales_train = sales_train[sales_train.item_price <= 100000]
sales_train = sales_train[sales_train.item_cnt_day <= 1000]

# Adjusting negatice prices (change it for median values)
median = sales_train[(sales_train.shop_id == 32) & (sales_train.item_id == 2973) & (sales_train.date_block_num == 4) & (sales_train.item_price > 0)].item_price.median()
sales_train.loc[sales_train.item_price < 0, 'item_price'] = median
# Якутск Орджоникидзе, 56
sales_train.loc[sales_train.shop_id == 0, 'shop_id'] = 57
testd.loc[testd.shop_id == 0, 'shop_id'] = 57
# Якутск ТЦ "Центральный"
sales_train.loc[sales_train.shop_id == 1, 'shop_id'] = 58
testd.loc[testd.shop_id == 1, 'shop_id'] = 58
# Жуковский ул. Чкалова 39м²
sales_train.loc[sales_train.shop_id == 10, 'shop_id'] = 11
testd.loc[testd.shop_id == 10, 'shop_id'] = 11
# РостовНаДону ТРК "Мегацентр Горизонт"
sales_train.loc[sales_train.shop_id == 39, 'shop_id'] = 40
testd.loc[testd.shop_id == 39, 'shop_id'] = 40
shop.shop_name.unique()
shop.loc[shop.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'
shop['shop_category'] = shop['shop_name'].str.split(' ').map(lambda x:x[1]).astype(str)
categories = ['Орджоникидзе,', 'ТЦ', 'ТРК', 'ТРЦ','ул.', 'Магазин', 'ТК', 'склад']
shop.shop_category = shop.shop_category.apply(lambda x: x if (x in categories) else 'etc')
shop.shop_category.unique()
shop.groupby(['shop_category']).sum()
from sklearn.preprocessing import LabelEncoder
category = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК']
shop.shop_category = shop.shop_category.apply(lambda x: x if (x in category) else 'etc')
print('Category Distribution', shop.groupby(['shop_category']).sum())

shop['shop_category_code'] = LabelEncoder().fit_transform(shop['shop_category'])
shop['city'] = shop['shop_name'].str.split(' ').map(lambda x: x[0])
shop.loc[shop.city == '!Якутск', 'city'] = 'Якутск'
shop['city_code'] = LabelEncoder().fit_transform(shop['city'])
shop = shop[['shop_id','city_code', 'shop_category_code']]

shop.head()
print(len(item_categories.item_category_name.unique()))
item_categories.item_category_name.unique()
item_categories['type'] = item_categories.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
item_categories.loc[(item_categories.type == 'Игровые') | (item_categories.type == 'Аксессуары'), 'category'] = 'Игры'
item_categories.loc[item_categories.type == 'PC', 'category'] = 'Музыка'
category = ['Игры', 'Карты', 'Кино', 'Книги','Музыка', 'Подарки', 'Программы', 'Служебные', 'Чистые', 'Аксессуары']
item_categories['type'] = item_categories.type.apply(lambda x: x if (x in category) else 'etc')
print(item_categories.groupby(['type']).sum())
item_categories['type_code'] = LabelEncoder().fit_transform(item_categories['type'])

# if subtype is nan then type
item_categories['split'] = item_categories.item_category_name.apply(lambda x: x.split('-'))
item_categories['subtype'] = item_categories['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
item_categories['subtype_code'] = LabelEncoder().fit_transform(item_categories['subtype'])
item_categories = item_categories[['item_category_id','type_code', 'subtype_code']]

item_categories.head()
sales_train['date']

sales_train['date'] = pd.to_datetime(sales_train['date'], format='%d.%m.%Y')
sales_train['month'] = sales_train['date'].dt.month
sales_train['year'] = sales_train['date'].dt.year
sales_train = sales_train.drop(columns=['date'])

# sales.head()
to_append = testd[['shop_id', 'item_id']].copy()

to_append['date_block_num'] = sales_train['date_block_num'].max() + 1
to_append['year'] = 2015
to_append['month'] = 11
to_append['item_cnt_day'] = 0
to_append['item_price'] = 0

sales_train = pd.concat([sales_train, to_append], ignore_index=True, sort=False)
sales_train.head()

period = sales_train[['date_block_num', 'year', 'month']].drop_duplicates().reset_index(drop=True)
period['days'] = period.apply(lambda r: monthrange(r.year, r.month)[1], axis=1)

sales_train = sales_train.drop(columns=['month', 'year'])

period.head()

from itertools import product
index_cols = ['date_block_num', 'shop_id', 'item_id']
grid = [] 
for block_num in sales_train['date_block_num'].unique():
    cur_shops = sales_train.loc[sales_train['date_block_num'] == block_num, 'shop_id'].unique()
    cur_items = sales_train.loc[sales_train['date_block_num'] == block_num, 'item_id'].unique()
    grid.append(np.array(list(product(*[[block_num], cur_shops, cur_items])), dtype='int16'))

# Turn the grid into a dataframe
grid = pd.DataFrame(np.vstack(grid), columns = index_cols, dtype = np.int16)
grid.head()
data = pd.merge(grid, shop, on='shop_id')
data = pd.merge(data, item, on='item_id')
data = pd.merge(data, item_categories, on='item_category_id')
data = pd.merge(data, period, on='date_block_num')

data
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

%matplotlib inline
type(data)
data1 = data[['date_block_num', 'year', 'month','days']]# 'item_price', 'item_cnt_day'

# Adjusting columns order
data = data[['date_block_num', 'year', 'month', 'days', 'city_code', 'shop_category_code', 'shop_id', 'item_category_id', 'type_code', 'subtype_code', 'item_id']] # 'item_price', 'item_cnt_day'

# Downcasting values
for c in ['date_block_num', 'month', 'days', 'city_code', 'shop_category_code', 'shop_id', 'item_category_id', 'type_code', 'subtype_code']:
    data[c] = data[c].astype(np.int8)
data['item_id'] = data['item_id'].astype(np.int16)
data['year'] = data['year'].astype(np.int16)

# Remove unused and temporary datasets
del grid, shop, item, item_categories, to_append

data.head()
aux = sales_train\
.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)\
.agg({'item_cnt_day' : 'sum', 'item_price' : 'mean'})\
.rename(columns= {'item_cnt_day' : 'item_cnt_month', 'item_price' : 'item_price_month'})

aux['item_cnt_month'] = aux['item_cnt_month'].astype(np.float16)
aux['item_price_month'] = aux['item_price_month'].astype(np.float16)

month_summary = pd.merge(data, aux, how='left', on=['date_block_num', 'shop_id', 'item_id'])\
    .fillna(0.0).sort_values(by=['shop_id', 'item_id', 'date_block_num'])

del data, aux

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
month_summary = agg_by_and_lag(month_summary, ['date_block_num'], 'date_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_id'], 'date_item_avg_item_cnt', [1,2,3,6,12])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code'], 'date_city_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id'], 'date_shop_avg_item_cnt', [1,2,3,6,12])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_category_id'], 'date_cat_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'type_code'], 'date_type_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'subtype_code'], 'date_subtype_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code'], 'date_shop_category_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'item_category_id'], 'date_shop_cat_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'type_code'], 'date_shop_type_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'subtype_code'], 'date_shop_subtype_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code', 'subtype_code'], 'date_shop_category_subtype_avg_item_cnt', [1])

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code', 'item_id'], 'date_item_city_avg_item_cnt', [1])
month_summary = agg_by_and_lag(month_summary, ['date_block_num'], 'date_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_id'], 'date_item_avg_item_price', [1,2,3,6,12], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code'], 'date_city_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id'], 'date_shop_avg_item_price', [1,2,3,6,12], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'item_category_id'], 'date_cat_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'type_code'], 'date_type_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'subtype_code'], 'date_subtype_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code'], 'date_shop_category_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'item_category_id'], 'date_shop_cat_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'type_code'], 'date_shop_type_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_id', 'subtype_code'], 'date_shop_subtype_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'shop_category_code', 'subtype_code'], 'date_shop_category_subtype_avg_item_price', [1], 'item_price_month')

month_summary = agg_by_and_lag(month_summary, ['date_block_num', 'city_code', 'item_id'], 'date_item_city_avg_item_price', [1], 'item_price_month')
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

with open('./gbm_model.pickle', 'wb') as handle:
    pickle.dump(gbm_model, handle)
y_pred = gbm_model.predict(X_test).clip(0, 20)

result = pd.merge(testd, X_test.assign(item_cnt_month=y_pred), how='left', on=['shop_id', 'item_id'])[['ID', 'item_cnt_month']]
result.to_csv('submission.csv', index=False)
