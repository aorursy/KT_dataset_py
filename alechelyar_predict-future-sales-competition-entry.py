import os

import re

import gc

import math

import numpy as np

import pandas as pd

import datetime

from itertools import product

from tqdm import tqdm_notebook as tqdm



import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

%matplotlib inline



import lightgbm as lgb

import xgboost as xgb

from sklearn.preprocessing import LabelEncoder



import stldecompose

import statsmodels.formula.api as smf

import statsmodels.tsa.api as smt

import statsmodels.api as sm



print(os.listdir('../input'))
sales = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

sales['date'] = sales.date.apply(lambda x: datetime.datetime.strptime(x, '%d.%m.%Y'))

categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

sales.head()
len(sales.date.apply(lambda x: x.strftime('%m-%d')).unique())
sales.date_block_num.unique()
def plot_seasonal(res, axes, axes_col):

    for i, p in enumerate(['observed', 'trend', 'seasonal', 'resid']):

        axes[i, axes_col].plot(getattr(res, p))

        if not axes_col:

            axes[i, axes_col].set_ylabel(p.title())
total_ts = sales.groupby('date_block_num').item_cnt_day.sum()

fig, axes = plt.subplots(4, 3)



add_decomposed = sm.tsa.seasonal_decompose(total_ts.values, freq=12, model="additive")

mul_decomposed = sm.tsa.seasonal_decompose(total_ts.values, freq=12, model="multiplicative")

stl_decompose = stldecompose.decompose(total_ts.values, period=12)



plot_seasonal(add_decomposed, axes, 0)

plot_seasonal(mul_decomposed, axes, 1)

plot_seasonal(stl_decompose, axes, 2)



axes[0, 0].set_title('Additive')

axes[0, 1].set_title('Multiplicative')

axes[0, 2].set_title('Lowess')

fig.tight_layout();
def plot_monthly_shop_sales(shops_per_row=3, height_scalar=15):

    shop_sales = sales.groupby(['date', 'shop_id']).agg({'item_cnt_day': 'sum'})

    shop_sales = shop_sales.unstack(level=0).transpose()

    

    num_shops = len(shop_sales.columns)

    

    nrows = math.ceil(num_shops / shops_per_row)

    height = height_scalar * shops_per_row

    

    fig, axes = plt.subplots(nrows, 1, figsize=(10, height))

    for i in range(0, num_shops, shops_per_row):

        ax_row = axes[int(i / shops_per_row)]

        shop_sales.iloc[:,i:i+shops_per_row].plot(ax=ax_row, alpha=0.8)

        ax_row.set_xlabel('')

    fig.tight_layout()

plot_monthly_shop_sales()
shop_months = [{'shop_id': ind, 'total_months': len(x), 'min': min(x), 

                'max': max(x), 'missing_months': 1 + max(x) - min(x) != len(x)} 

               for ind, x in 

                   sales.groupby(['shop_id']).date_block_num.unique().items()]

shop_months = pd.DataFrame(shop_months)

shop_months[shop_months['max'] < 33]
items = pd.read_csv('../input/items.csv')

items.tail()
categories = pd.read_csv('../input/competitive-data-science-predict-future-sales/item_categories.csv')

categories.head()
categories[categories.item_category_name.apply(lambda x: ' - ' not in x and '(' not in x)]
def get_main_cat(name):

    if ' - ' in name:

        return name.split(' - ')[0]

    elif '(' in name:

        return name.split('(')[0].strip()

    return name

categories['main_category'] = categories.item_category_name.apply(get_main_cat)

categories.main_category.unique()
translate_categories = {

    'PC': 'PC',

    'Аксессуары': 'Accessories',

    'Цифра': 'Figure',

    'Доставка товара': 'Delivery of goods',

    'Игровые консоли': 'Game consoles',

    'Игры': 'Games',

    'Игры Android': 'Android games',

    'Игры MAC': 'Games MAC',

    'Игры PC': 'Games PC',

    'Кино, Музыка, Игры': 'Cinema, Music, Games',

    'Карты оплаты': 'Payment cards',

    'Кино': 'Cinema',

    'Билеты': 'Tickets',

    'Книги': 'Books',

    'Музыка': 'Music',

    'Подарки': 'Gifts',

    'Программы': 'Programs',

    'Служебные': 'Utilities',

    'Чистые носители': 'Clean Media',

    'Элементы питания': 'Batteries'

}

categories['main_category'] = categories.main_category.apply(lambda x: translate_categories[x])

categories.main_category.unique()
shops = pd.read_csv('../input/competitive-data-science-predict-future-sales/shops.csv').set_index('shop_id')

shops.tail()
_shop_name_replace = {

    'Выездная Торговля': 'Выездная Торговля ""',

    'Жуковский ул. Чкалова 39м?': 'Жуковский ул. "Чкалова 39м?"',

    'Жуковский ул. Чкалова 39м²': 'Жуковский ул. "Чкалова 39м²"',

    'Воронеж (Плехановская, 13)': 'Воронеж "Плехановская, 13"',

    'Интернет-магазин ЧС': 'Интернет-магазин "ЧС"',

    'Москва Магазин С21': 'Москва "Магазин С21"',

    'Цифровой склад 1С-Онлайн': 'Цифровой склад "1С-Онлайн"',

    'Якутск Орджоникидзе, 56': 'Якутск "Орджоникидзе, 56"',

    '!Якутск Орджоникидзе, 56 фран': 'Якутск "Орджоникидзе, 56 фран"',

    'Воронеж ТРЦ Сити-Парк "Град"': 'Воронеж ТРЦ "Сити-Парк Град"'

}

shops['shop_name'] = shops.shop_name.apply(lambda x: _shop_name_replace[x] if x in _shop_name_replace else x)
_shop_type = {

    'ТЦ': 'Shopping center',

    'ТРК': 'Dispenser',

    'ТРЦ': 'Shopping mall',

    'ТК': 'TC',

    'МТРЦ': 'MTRC',

    'Цифровой': 'Digital Warehouse',

    'Интернет-магазин': 'Online'

}

#Get the shop type of the store

_type = shops.shop_name.apply(lambda x: [w for w in x.split() if w in _shop_type])

shops['shop_type'] = _type.apply(lambda x: _shop_type[x[0]] if len(x) else np.nan)

#Get the city from the shop name: !Moscow TC "Store 26" -> Moscow

_city = shops.shop_name.apply(lambda x: x.split(' "')[0])

_city = _city.apply(lambda x: " ".join([w for w in x.split() if w not in _shop_type]))

_city = _city.str.replace('!', '').str.title()

shops['shop_city'] = _city

#Get the 

shops.head()
_pop_replace = {

    '': np.nan, 'Адыгея': 282419, 'Балашиха': 228567, 'Волжский': 320761, 'Вологда': 305397, 

    'Воронеж': 997447, 'Выездная Торговля': np.nan, 'Жуковский Ул.': 107994, 'Казань': 1169000, 

    'Калуга': 328871, 'Коломна': 144838, 'Красноярск': 1007000, 'Курск': 425950, 

    'Москва': 11920000, 'Мытищи': 176825, 'Н.Новгород': 1257000, 'Новосибирск': 1511000, 

    'Омск': 1159000, 'Ростовнадону': 1100000, 'Спб': 4991000, 'Самара': 1170000, 

    'Сергиев Посад': 109076, 'Сургут': 321062, 'Томск': 543596, 'Тюмень': 621918, 'Якутск': 282419, 

    'Уфа': 1075000, 'Химки': 218275, 'Склад': np.nan, 'Чехов': 71301, 'Ярославль': 597161

}

shops['shop_city_pop'] = shops.shop_city.map(lambda x: _pop_replace[x])

shops.head()
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')

test.head()
len(set(test.item_id) - set(sales.item_id)), len(set(test.shop_id) - set(sales.shop_id))
test.shop_id.nunique() * test.item_id.nunique() == test.shape[0]
#Create index for every combination of shop and item per month

index = []

for dbn in sales.date_block_num.unique():

    shop_ids = sales[sales.date_block_num == dbn].shop_id.unique()

    item_ids = sales[sales.date_block_num == dbn].item_id.unique()

    index.append(np.array(list(product(shop_ids, item_ids, [dbn])), dtype='int16'))

index = pd.DataFrame(np.vstack(index), columns=['shop_id', 'item_id', 'date_block_num'])

index = index.set_index(['shop_id', 'item_id', 'date_block_num']).index

index.shape
sales_train = sales.groupby(['shop_id', 'item_id', 'date_block_num'])

sales_train = sales_train.agg({'item_cnt_day':'sum'})

sales_train = sales_train.reindex(index=index)

sales_train = sales_train.reset_index()

sales_train = sales_train.rename(columns={'item_cnt_day': 'item_cnt_month'})

sales_train['item_cnt_month'] = sales_train.item_cnt_month.clip(0, 20).fillna(0)

sales_train = sales_train.set_index(['shop_id', 'item_id', 'date_block_num'])

sales_train.shape
sales_train = sales_train.reset_index().merge(items[['item_id', 'item_category_id']], on = 'item_id')

sales_train.head()
sales_train = sales_train.fillna(0)

train_X, train_y = sales_train.drop('item_cnt_month', axis=1), sales_train.item_cnt_month.clip(0, 20)

model = lgb.LGBMModel(objective='regression', max_depth=10, n_estimators=100, min_child_weight=0.5, 

                         random_state=40, n_jobs=-1, silent=False)

model.fit(train_X, train_y, eval_metric='rmse')
test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

test = test.merge(items[['item_id', 'item_category_id']], on='item_id')

test = test.set_index('ID')

test['date_block_num'] = 34

pred = model.predict(test)

pred = pd.DataFrame(pred, columns=['item_cnt_month']).fillna(0).clip(0, 20)

pred.index.names = ['ID']

pred.to_csv('baseline.csv')
def set_type(series, to_float=False):

    ints = {'int8':255, 'int16':65535, 'int32':2147483647, 'int64':np.inf}

    floats = {'float16':32767, 'float32':2147483647, 'float64':np.inf}

    dtype = series.dtype.name

    if dtype in ints and not to_float:

        maxval = series.abs().max()

        for key, val in ints.items():

            if maxval < val:

                return series.astype(key)

    if dtype in floats or (to_float and dtype.startswith('int')):

        maxval = series.abs().max()

        for key, val in floats.items():

            if maxval < val:

                return series.astype(key)

    if dtype in {'object', 'category'}:

        l = LabelEncoder()

        return l.fit_transform(series).astype('int8')

    return series

def minimize_memory(df, to_float=False):

    df = df.reset_index()

    for col in df.columns:

        df[col] = set_type(df[col], to_float)

    return df
del add_decomposed, mul_decomposed, stl_decompose, shop_months, train_X, train_y, model, pred, shop_ids, item_ids

gc.collect()
lags = [1, 2, 3, 6, 12]

#train

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

train = train[train.item_price < 100000]

train = train[train.item_cnt_day <= 1000]

train = pd.read_csv('../input/competitive-data-science-predict-future-sales/sales_train.csv')

median = train.loc[(train.shop_id==32) & (train.item_id==2973) & 

                   (train.date_block_num==4) & (train.item_price>0)].item_price.median()

train.loc[train.item_price < 0, 'item_price'] = median

train['item_revenue_month'] = train.item_cnt_day * train.item_price

train = train.groupby(['shop_id', 'item_id', 'date_block_num'])

train = train.agg({'item_cnt_day': 'sum', 'item_price': 'mean', 'item_revenue_month': 'sum'})

train.columns = ['item_cnt_month', 'item_mean_price', 'item_revenue_month']

train = train.reindex(index=index)

#test

test = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv')

test_ids = test.ID

test['date_block_num'] = 34

test = test.set_index(['shop_id', 'item_id', 'date_block_num']).drop('ID', axis=1)

#join together

train_test = pd.concat([train, test])

train_test = train_test.fillna(0).astype({'item_cnt_month': 'int32', 'item_mean_price': 'int32'})

train_test = minimize_memory(train_test)

train_test.head()
#Remove outliers

train_test.loc[train_test.shop_id == 0, 'shop_id'] = 57

train_test.loc[train_test.shop_id == 1, 'shop_id'] = 58

train_test.loc[train_test.shop_id == 10, 'shop_id'] = 11

train_test['item_cnt_month'] = train_test.item_cnt_month.fillna(0).clip(0, 20)

train_test.set_index(['shop_id', 'item_id', 'date_block_num'], inplace=True)
def add_aggregate_lags(df, gb_cols, target_col, prefix, astype='float64', 

                       fillna=np.nan, lags=lags, agg='mean'):

    df = df.reset_index()

    _gb = df.groupby(gb_cols).agg({target_col: agg})

    for lag in lags:

        _temp = _gb.copy()

        name = prefix + str(lag)

        _temp.reset_index(inplace=True)

        _temp['date_block_num'] += lag

        _temp = _temp.rename(columns={target_col: name})

        df = pd.merge(df, _temp, on=gb_cols, how='left')

        df[name] = df[name].fillna(fillna).astype(astype)

    return df.set_index(['shop_id', 'item_id', 'date_block_num'])

train_test = add_aggregate_lags(train_test, ['shop_id', 'item_id', 'date_block_num'], 'item_cnt_month', 

                                'shop_item_sales_lag_', fillna=0, astype='int16', agg='sum')

train_test = add_aggregate_lags(train_test, ['shop_id', 'date_block_num'], 'item_cnt_month', 'shop_sales_lag_', 

                                fillna=0, astype='int32', agg='sum')

train_test = add_aggregate_lags(train_test, ['item_id', 'date_block_num'], 'item_cnt_month', 'item_sales_lag_', 

                                fillna=0, astype='int32', agg='sum')

train_test = add_aggregate_lags(train_test, ['shop_id', 'item_id', 'date_block_num'], 'item_mean_price', 

                                'shop_item_price_lag_', lags=[1, 2, 3])

train_test = add_aggregate_lags(train_test, ['item_id', 'date_block_num'], 'item_mean_price', 

                                'item_price_lag_', lags=[1, 2, 3])

train_test.tail()
days_per_block = pd.DataFrame({'days_per_block': sales.date.apply(lambda x: x.strftime('%m-%d'))})

days_per_block['date_block_num'] = sales.date_block_num

days_per_block = days_per_block.groupby('date_block_num').nunique()[['days_per_block']].reset_index()

days_per_block = days_per_block.append({'date_block_num': 34, 'days_per_block': 30}, ignore_index=True)

train_test = train_test.reset_index().merge(days_per_block, on='date_block_num')

train_test['days_per_block'] = (train_test.days_per_block - 30).astype('int8')

train_test.set_index(['shop_id', 'item_id', 'date_block_num'], inplace=True)

train_test[['days_per_block']].tail()
#Inspired by dlarionov

def add_since_features(df, key_func, name):

    known = {}

    df[name] = -1

    df[name] = df[name].astype(np.int8)

    for i, row in df.iterrows():    

        key = key_func(row)

        if key not in known:

            if row.item_cnt_month > 0:

                known[key] = row.date_block_num

        else:

            if known[key] < row.date_block_num:

                df.at[i, name] = row.date_block_num - known[key]

                known[key] = row.date_block_num  

    return df

train_test.reset_index(inplace=True)

train_test = add_since_features(train_test, lambda r: str(r.shop_id) + '_' + str(r.item_id), 'm_since_last_shop_item_sale')

train_test = add_since_features(train_test, lambda r: str(r.shop_id), 'm_since_last_shop_sale')

train_test = add_since_features(train_test, lambda r: str(r.item_id), 'm_since_last_item_sale')

train_test.set_index(['shop_id', 'item_id', 'date_block_num'], inplace=True)

train_test.iloc[-5:,-3:]
def add_mean_encodings(df, index_cols, target_col, name):

    _gb = df.groupby(index_cols)[[target_col]].mean()

    _gb.rename(columns={target_col: name}, inplace=True)

    return pd.merge(df, _gb.reset_index(), on=index_cols)

train_test.reset_index(inplace=True)

train_test = add_mean_encodings(train_test, ['shop_id'], 'item_cnt_month', 'shop_mean')

train_test = add_mean_encodings(train_test, ['item_id'], 'item_cnt_month', 'item_mean')

train_test.set_index(['shop_id', 'item_id', 'date_block_num'], inplace=True)

train_test = minimize_memory(train_test).set_index(['shop_id', 'item_id', 'date_block_num'])

train_test[['shop_mean', 'item_mean']].tail()
_gb = sales.groupby(['shop_id', 'item_id', 'date_block_num']).agg({'item_cnt_day': ['sum', 'last']})

_gb = _gb.join(train_test[['days_per_block']], how='left')

_gb.columns = ['month_sum', 'month_last', 'days_per_block']

_gb['end_of_month_percent'] = _gb.month_last / (_gb.month_sum / (_gb.days_per_block + 30))

train_test = train_test.join(_gb[['end_of_month_percent']])

train_test[['end_of_month_percent']].head()
train_test.reset_index(inplace=True)

train_test = pd.merge(train_test, shops[['shop_type', 'shop_city', 'shop_city_pop']].reset_index(), 

         on='shop_id', validate='many_to_one')

train_test = add_mean_encodings(train_test, ['shop_type'], 'item_cnt_month', 'shop_type_mean')

train_test = add_mean_encodings(train_test, ['shop_city'], 'item_cnt_month', 'shop_city_mean')

train_test.set_index(['shop_id', 'item_id', 'date_block_num'], inplace=True)

train_test[['shop_type', 'shop_city', 'shop_city_pop']].iloc[20000:20010,:]
train_test.reset_index(inplace=True)

items = items.merge(categories, on='item_category_id', validate='m:1', how='left')

train_test = train_test.merge(items[['item_id', 'item_category_name', 'main_category']], 

                              on='item_id', how='left', validate='m:1')

train_test = train_test.rename(columns={'item_category_name': 'item_category_full', 'main_category': 'item_category_main'})

train_test = add_mean_encodings(train_test, ['item_category_full'], 'item_cnt_month', 'item_category_full_mean')

train_test = add_mean_encodings(train_test, ['item_category_main'], 'item_cnt_month', 'item_category_main_mean')

train_test.set_index(['shop_id', 'item_id', 'date_block_num'], inplace=True)

train_test = minimize_memory(train_test).set_index(['shop_id', 'item_id', 'date_block_num'])

train_test.iloc[:5,-4:]
train_test = add_aggregate_lags(train_test, ['date_block_num'], 'item_revenue_month', 'monthly_revenue_lag_', 

                   astype='int32', fillna=0, agg='sum', lags=[1, 2, 3, 6, 12])

train_test = add_aggregate_lags(train_test, ['shop_id', 'item_id', 'date_block_num'], 'item_revenue_month', 

                   'shop_item_revenue_lag_', astype='int32', fillna=0, agg='sum', lags=[1, 2, 3, 6, 12])

train_test.iloc[:5,-2:]
train_test = add_aggregate_lags(train_test, ['date_block_num'], 'item_cnt_month', 

                   'monthly_sales_lag', astype='int16', fillna=0, agg='sum', lags=[1, 2, 3, 6, 12])
#Inspired by dlarionov

train_test.reset_index(inplace=True)

train_test['m_since_shop_item_first_sale'] = train_test['date_block_num'] - train_test.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

train_test['m_since_item_first_sale'] = train_test['date_block_num'] - train_test.groupby('item_id')['date_block_num'].transform('min')
train_test['month'] = train_test.date_block_num % 12

train_test.set_index(['shop_id', 'item_id', 'date_block_num'], inplace=True)
train_test = minimize_memory(train_test).set_index(['shop_id', 'item_id', 'date_block_num'])

train_test.to_pickle('data.pkl')

del _gb, sales, items, categories, test, train, shops

gc.collect();
train_test = minimize_memory(pd.read_pickle('../input/traintestset/data.pkl'), to_float=False)

train_test.info()
train_test.drop(['item_mean_price', 'item_revenue_month'], axis=1, inplace=True) #Only used for lag features

train_test.reset_index(inplace=True)

X_train = train_test[(12 <= train_test.date_block_num) & (train_test.date_block_num <= 32)].drop('item_cnt_month', axis=1)

y_train = train_test[(12 <= train_test.date_block_num) & (train_test.date_block_num <= 32)].item_cnt_month

X_eval = train_test[train_test.date_block_num == 33].drop('item_cnt_month', axis=1)

y_eval = train_test[train_test.date_block_num == 33].item_cnt_month

del train_test

gc.collect()
xgb_model = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=1000,

                             verbosity=1, booster='gbtree', gamma=0, random_state=40,

                             min_child_weight=1, reg_alpha=0, reg_lambda=1)

#xgb_model.fit(X_train, y_train, eval_metric='rmse', early_stopping_rounds=50,

#              eval_set=[(X_train, y_train), (X_eval, y_eval)])
categoricals = "name:shop_type,shop_city,item_category_full,item_category_main"

lgb_model = lgb.LGBMRegressor(boosting_type='gbdt', num_leaves=31, objective='regression_l2',

                                max_depth=-1, learning_rate=0.1, 

                                n_estimators=1000, reg_alpha=0.0, reg_lambda=0.0, 

                                random_state=40, n_jobs=-1, silent=True, categorical_feature=categoricals)

#lgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_eval, y_eval)], 

#              verbose=True, eval_names=['Test', 'Eval'], early_stopping_rounds=50)
lgb_results = pd.DataFrame({'Eval': lgb_model.evals_result_['Eval']['l2'], 'Test': lgb_model.evals_result_['Test']['l2']})

lgb_results.plot(figsize=(15, 10))