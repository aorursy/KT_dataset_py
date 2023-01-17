import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

from scipy import stats

import logging

import datetime



from sklearn.preprocessing import OneHotEncoder,MinMaxScaler, Normalizer, LabelEncoder

from sklearn.feature_selection import SelectKBest,chi2,SelectFromModel

from xgboost import XGBClassifier, XGBRegressor

from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.model_selection import GridSearchCV

from sklearn.impute import SimpleImputer

from xgboost import plot_importance

from mlxtend.preprocessing import DenseTransformer

from mlxtend.feature_selection import ColumnSelector

from itertools import product



sns.set(color_codes=True)
def _transfer_type(df, cols, dtype):

    for v in cols:

        df[v] = df[v].astype(dtype)

    

    return df
dateparse = lambda x: pd.datetime.strptime(x, '%d.%m.%Y')

sales_df = pd.read_csv('../input/sales_train.csv', parse_dates = ['date'], date_parser=dateparse)



item_df = pd.read_csv('../input/items.csv')

shop_df = pd.read_csv('../input/shops.csv')

category_df = pd.read_csv('../input/item_categories.csv')



test_df = pd.read_csv('../input/test.csv').set_index('ID')
sales_df = _transfer_type(sales_df, ['date_block_num', 'shop_id', 'item_id', 'item_cnt_day'], np.int16)

sales_df = _transfer_type(sales_df, ['item_price'], np.float16)



item_df = _transfer_type(item_df, ['item_id', 'item_category_id'], np.int16)

item_df['item_name'] = item_df['item_name'].astype(str)



shop_df['shop_name'] = shop_df['shop_name'].astype(str)

shop_df['shop_id'] = shop_df['shop_id'].astype(np.int16)



category_df['item_category_name'] = category_df['item_category_name'].astype(str)

category_df['item_category_id'] = category_df['item_category_id'].astype(np.int16)
sales_df = sales_df[sales_df.item_price<100000]

sales_df = sales_df[sales_df.item_cnt_day<1000]
sales_df[sales_df.item_price < 0]
sales_df.at[484683, 'item_price'] = sales_df[(sales_df.item_id == 2973) & (sales_df.item_price > 0)].item_price.mean()
shop_df[shop_df.shop_id == 0]
shop_df[shop_df.shop_id == 57]
# Якутск Орджоникидзе, 56

sales_df.at[sales_df.shop_id == 0, 'shop_id'] = 57

# Якутск ТЦ "Центральный"

sales_df.at[sales_df.shop_id == 1, 'shop_id'] = 58

# Жуковский ул. Чкалова 39м²

sales_df.at[sales_df.shop_id == 10, 'shop_id'] = 11
def _rename(prefix):

    cols = ['2013-01', '2013-02', '2013-03', '2013-04','2013-05','2013-06','2013-07','2013-08','2013-09','2013-10','2013-11','2013-12','2014-01','2014-02','2014-03','2014-04','2014-05','2014-06','2014-07','2014-08','2014-09','2014-10','2014-11','2014-12','2015-01','2015-02','2015-03','2015-04','2015-05','2015-06','2015-07','2015-08', '2015-09','2015-10']

    

    result = {}

    for i in range(1, len(cols) + 1, 1):

        result[cols[i-1]] = prefix + str(i)

    

    return result
import gc

t = sales_df.groupby([sales_df.date.apply(lambda x: x.strftime('%Y-%m')),'item_id','shop_id']).agg({'item_cnt_day': 'sum'}).reset_index()



t = t[['date','item_id','shop_id','item_cnt_day']]

t = t.pivot_table(index=['item_id','shop_id'], columns='date',values='item_cnt_day',fill_value=0).reset_index()

t = t.rename(index=str, columns=_rename('r'))

sales_detail_df = t.copy()



del t

gc.collect()
leak_df = test_df[['item_id', 'shop_id']].drop_duplicates()

sales_detail_df = pd.merge(sales_detail_df, leak_df, on=['item_id', 'shop_id'], how='outer')

sales_detail_df = sales_detail_df.fillna(0)

sales_detail_df.info()
def _extract(df, i, is_test=False):

    arr = ['item_id','shop_id']

    for j in range(1, 13, 1):

        arr = arr + ['r' + str(i-j)]

    

    if not is_test:

        arr = arr + ['r' + str(i)]

    

    tdf = df[arr]

    tdf['date_block_num'] = i - 1

    

    return tdf
cols = ['item_id','shop_id', 'r1', 'r2', 'r3', 'r4',  'r5', 'r6','r7', 'r8',  'r9', 'r10',  'r11',  'r12', 'item_cnt_month', 'date_block_num']

sales_record_df = pd.DataFrame(columns=cols)

for i in range(13, 35, 1):

    tdf = _extract(sales_detail_df, i)

    sales_record_df = sales_record_df.append(pd.DataFrame(columns=cols, data=tdf.values))



sales_record_df['month'] = sales_record_df['date_block_num'].apply(lambda x: (x % 12) + 1)



for i in range(1, 13, 1):

    sales_record_df['r'+str(i)] = sales_record_df['r'+str(i)].astype(np.float16)



sales_record_df = _transfer_type(sales_record_df, ['item_id', 'shop_id', 'date_block_num', 'month'], np.int16)

sales_record_df = _transfer_type(sales_record_df, ['item_cnt_month'], np.float16)

sales_record_df = sales_record_df.fillna(0)
shop_df.loc[shop_df.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shop_df['shop_name'] = shop_df['shop_name'].astype(str)

shop_df['city'] = shop_df['shop_name'].str.split(' ').map(lambda x: x[0])

shop_df.loc[shop_df.city == '!Якутск', 'city'] = 'Якутск'



encoder = LabelEncoder()

encoder.fit(shop_df['city'])

shop_df['city_code'] = encoder.transform(shop_df['city'])
category_df['item_category_name'] = category_df['item_category_name'].astype(str)

category_df['type'] = category_df['item_category_name'].map(lambda x: x.split('-')[0].strip())



encoder = LabelEncoder()

encoder.fit(category_df['type'])

category_df['type_code'] = encoder.transform(category_df['type'])
category_df['subtype'] = category_df['item_category_name'].map(lambda x: x.split('-')[1].strip() if len(x.split('-')) > 1 else x.split('-')[0].strip())



encoder = LabelEncoder()

encoder.fit(category_df['subtype'])

category_df['subtype_code'] = encoder.transform(category_df['subtype'])
sales_record_df = pd.merge(sales_record_df, item_df, on=['item_id'], how='left')

sales_record_df = pd.merge(sales_record_df, shop_df, on=['shop_id'], how='left')

sales_record_df = pd.merge(sales_record_df, category_df, on=['item_category_id'], how='left')

sales_record_df = sales_record_df.fillna(0)



sales_record_df = sales_record_df.drop(['city', 'type', 'subtype', 'item_name', 'shop_name', 'item_category_name'], axis=1)

sales_record_df = _transfer_type(sales_record_df, ['item_category_id', 'city_code', 'type_code', 'subtype_code'], np.int16)



sales_record_df.info()
def _agg(df, cols, prefix):

    

    result = df[cols].drop_duplicates()

    

    for i in range(1, 13, 1):

        t = df.groupby(cols)['r'+str(i)].mean().fillna(0).astype(np.float16).reset_index(name=prefix+str(i))

        result = pd.merge(result, t, on=cols, how='left')

    

#     result = _mean(result, prefix)

    return result
sc_df = _agg(sales_record_df, ['shop_id', 'item_category_id'], 'sc')

i_df = _agg(sales_record_df, ['item_id'], 'i')

it_df = _agg(sales_record_df, ['item_category_id'], 'it')

s_df = _agg(sales_record_df, ['shop_id'], 's')



sales_record_df = pd.merge(sales_record_df, sc_df, on=['shop_id', 'item_category_id'], how='left')

sales_record_df = pd.merge(sales_record_df, i_df, on=['item_id'], how='left')

sales_record_df = pd.merge(sales_record_df, it_df, on=['item_category_id'], how='left')

sales_record_df = pd.merge(sales_record_df, s_df, on=['shop_id'], how='left')
dataset = sales_record_df.copy()

dataset.columns.values
dataset_beta = dataset[dataset.date_block_num < 33]

dataset_alpha = dataset[dataset.date_block_num == 33]
features = [

    'item_id', 'shop_id', 'r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7',

    'r8', 'r9', 'r10', 'r11', 'r12',

    'date_block_num', 'month', 'item_category_id', 'city_code',

    'type_code', 'subtype_code',

    

    'sc1', 'sc2', 'sc3', 'sc4', 'sc5', 'sc6', 'sc7', 'sc8', 'sc9', 'sc10', 'sc11', 'sc12',

    'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12',

    'it1', 'it2', 'it3', 'it4', 'it5', 'it6', 'it7', 'it8', 'it9', 'it10', 'it11', 'it12',

    's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12',

]

label = 'item_cnt_month'
import xgboost as xgb



dataset_beta = dataset[(dataset.date_block_num < 24) & (dataset.date_block_num > 17)]

train_dataset_x = dataset_beta[features]

train_dataset_y = dataset_beta[label].values.ravel()

train_dmatrix = xgb.DMatrix(train_dataset_x, label=train_dataset_y)



dataset_test = dataset[(dataset.date_block_num == 33)]

test_dataset_x = dataset_test[features]

test_dataset_y = dataset_test[label].values.ravel()

test_dmatrix = xgb.DMatrix(test_dataset_x, label=test_dataset_y)



dataset_valid = dataset[(dataset.date_block_num == 32)]

valid_dataset_x = dataset_valid[features]

valid_dataset_y = dataset_valid[label].values.ravel()

valid_dmatrix = xgb.DMatrix(valid_dataset_x, label=valid_dataset_y)
from xgboost import XGBRegressor



watchlist = [(train_dmatrix, 'train'), (valid_dmatrix, 'validate')] 



params = {

  'booster': 'gbtree',

  'objective': 'reg:linear',

  'eta': 0.1,

  'gamma': 0.7,

  'min_child_weight': 7,

  'max_depth': 4,

  'subsample': 0.5,

  'colsample_bytree': 0.1,

  'nthread': 2,

  'silent': 0,

  'seed': 2019,

  "max_evals": 200,

}



watchlist = [(train_dmatrix, 'train'), (valid_dmatrix, 'validate')] 

bst = xgb.train(params, train_dmatrix, evals=watchlist, early_stopping_rounds=10, num_boost_round=190)
model = xgb.train(params, train_dmatrix, num_boost_round=bst.best_iteration)
from sklearn.metrics import mean_squared_error

from math import sqrt



t = pd.merge(test_df, dataset_test, on=['shop_id', 'item_id'], how='left')

test_dmatrix = xgb.DMatrix(t[features], label=t[label].values.ravel())



pred = model.predict(test_dmatrix)

sqrt(mean_squared_error(t[label], pred))
tdf = _extract(sales_detail_df, 35, is_test=True)

cols = ['item_id','shop_id', 'r1', 'r2', 'r3', 'r4',  'r5', 'r6','r7', 'r8',  'r9', 'r10',  'r11',  'r12', 'date_block_num']

dataset_predict_df = pd.DataFrame(columns=cols, data=tdf.values)

dataset_predict_df = dataset_predict_df.drop_duplicates(['item_id','shop_id'])



dataset_predict_df = pd.merge(test_df, dataset_predict_df, on=['item_id', 'shop_id'], how='left')

dataset_predict_df = dataset_predict_df.fillna(0)



dataset_predict_df = pd.merge(dataset_predict_df, item_df, on=['item_id'], how='left')

dataset_predict_df = pd.merge(dataset_predict_df, shop_df, on=['shop_id'], how='left')

dataset_predict_df = pd.merge(dataset_predict_df, category_df, on=['item_category_id'], how='left')



dataset_predict_df['month'] = dataset_predict_df['date_block_num'].apply(lambda x: (x % 12) + 1)



sc_df = _agg(dataset_predict_df, ['shop_id', 'item_category_id'], 'sc')

i_df = _agg(dataset_predict_df, ['item_id'], 'i')

it_df = _agg(dataset_predict_df, ['item_category_id'], 'it')

s_df = _agg(dataset_predict_df, ['shop_id'], 's')



dataset_predict_df = pd.merge(dataset_predict_df, sc_df, on=['shop_id', 'item_category_id'], how='left')

dataset_predict_df = pd.merge(dataset_predict_df, i_df, on=['item_id'], how='left')

dataset_predict_df = pd.merge(dataset_predict_df, it_df, on=['item_category_id'], how='left')

dataset_predict_df = pd.merge(dataset_predict_df, s_df, on=['shop_id'], how='left')



dataset_predict_df = dataset_predict_df.drop(['city', 'type', 'subtype', 'item_name', 'shop_name', 'item_category_name'], axis=1)

dataset_predict_df = _transfer_type(dataset_predict_df, ['item_category_id', 'city_code', 'type_code', 'subtype_code', 'month'], np.int16)
predict_dmatrix = xgb.DMatrix(dataset_predict_df[features])

pred = model.predict(predict_dmatrix)
submission = pd.DataFrame({

    "ID": test_df.index, 

    "item_cnt_month": pred.clip(0, 20)

})



submission.to_csv('submission.csv', index=False)