# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.stats import skew
from scipy.special import boxcox1p
from scipy.stats import boxcox_normmax
from sklearn.linear_model import ElasticNetCV, LassoCV, RidgeCV,Ridge,Lasso,ElasticNet
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor,RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression
from sklearn.svm import SVR, LinearSVC
from sklearn.pipeline import make_pipeline,Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

from mlxtend.regressor import StackingCVRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import lightgbm as lgb
import os
import re
import shap
import pandas_profiling as ppf
import seaborn as sns
import matplotlib.pyplot as plt

import re
from collections import Counter
from operator import itemgetter
import time
from itertools import product
import datetime as dt
import calendar
import gc

RANDOM_SEED = 42
train = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/sales_train.csv')
shops = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/shops.csv')
items = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/items.csv')
cats = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/item_categories.csv')
test = pd.read_csv('/kaggle/input/competitive-data-science-predict-future-sales/test.csv')
train
def summary_stats_table(data):

    missing_counts = pd.DataFrame(data.isnull().sum())
    missing_counts.columns = ['count_null']

    num_stats = data.select_dtypes(include=['int64','float64']).describe().loc[['count','min','max']].transpose()
    num_stats['dtype'] = data.select_dtypes(include=['int64','float64']).dtypes.tolist()

    non_num_stats = data.select_dtypes(exclude=['int64','float64']).describe().transpose()
    non_num_stats['dtype'] = data.select_dtypes(exclude=['int64','float64']).dtypes.tolist()
    non_num_stats = non_num_stats.rename(columns={"first": "min", "last": "max"})

    stats_merge = pd.concat([num_stats, non_num_stats], axis=0, join='outer', ignore_index=False, keys=None,
              levels=None, names=None, verify_integrity=False, copy=True).fillna("").sort_values('dtype')

    column_order = ['dtype', 'count', 'count_null','unique','min','max','top','freq']
    summary_stats = pd.merge(stats_merge, missing_counts, left_index=True, right_index=True)[column_order]
    return(summary_stats)
train = train[train.item_price<100000] # drop 1
train = train[train.item_cnt_day<1000] # drop 2

train = train[train.item_price > 0].reset_index(drop=True) # drop 1

train.loc[train.item_cnt_day < 0, 'item_cnt_day'] = 0

train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57

train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58

train.loc[train.shop_id == 11, 'shop_id'] = 10
test.loc[test.shop_id == 11, 'shop_id'] = 10

train.loc[train.shop_id == 40, 'shop_id'] = 39
test.loc[test.shop_id == 40, 'shop_id'] = 39
shops.loc[shops.shop_name == 'Сергиев Посад ТЦ "7Я"', 'shop_name'] = 'СергиевПосад ТЦ "7Я"'

shops['city'] = shops['shop_name'].str.split(' ').map(lambda x: x[0])
shops['category'] = shops['shop_name'].str.split(' ').map(lambda x:x[1]).astype(str)

shops.loc[shops.city == '!Якутск', 'city'] = 'Якутск'

category = ['ТЦ', 'ТРК', 'ТРЦ', 'ТК']
shops.category = shops.category.apply(lambda x: x if (x in category) else 'etc')
shops.groupby(['category']).sum()

shops['shop_city'] = shops.city
shops['shop_category'] = shops.category

shops['shop_city'] = LabelEncoder().fit_transform(shops['shop_city'])
shops['shop_category'] = LabelEncoder().fit_transform(shops['shop_category'])

shops = shops[['shop_id','shop_city', 'shop_category']]
shops.head()
cats['type_code'] = cats.item_category_name.apply(lambda x: x.split(' ')[0]).astype(str)
cats.loc[(cats.type_code == 'Игровые') | (cats.type_code == 'Аксессуары'), 'type_code'] = 'Игры'
cats.loc[cats.type_code == 'PC', 'type_code'] = 'Музыка'

category = ['Игры', 'Карты', 'Кино', 'Книги','Музыка', 'Подарки', 'Программы', 'Служебные', 'Чистые']
cats['type_code'] = cats.type_code.apply(lambda x: x if (x in category) else 'etc')
cats['type_code'] = LabelEncoder().fit_transform(cats['type_code'])

cats['split'] = cats.item_category_name.apply(lambda x: x.split('-'))
cats['subtype'] = cats['split'].map(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats['subtype_code'] = LabelEncoder().fit_transform(cats['subtype'])

cats = cats[['item_category_id','type_code', 'subtype_code']]
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

items['type'] = items.name_2.apply(lambda x: x[0:8] if x.split(' ')[0] == 'xbox' else x.split(' ')[0])
items.loc[(items.type == 'x360') | (items.type == 'xbox360'), 'type'] = 'xbox 360'
items.loc[items.type == '', 'type'] = 'mac'
items.type = items.type.apply(lambda x: x.replace(' ',''))
items.loc[(items.type == 'pc') | (items.type == 'pс') | (items.type == 'рс'), 'type'] = 'pc'
items.loc[(items.type == 'рs3'), 'type'] = 'ps3'

group_sum = items.groupby('type').sum()
drop_list = group_sum.loc[group_sum.item_category_id < 200].index

print('drop list:', drop_list)

items.name_2 = items.type.apply(lambda x: 'etc' if x in drop_list else x)
items = items.drop(['type'], axis=1)
print(items.groupby('name_2').count()[['item_id']])

items['name_2'] = LabelEncoder().fit_transform(items['name_2']).astype(np.int8)
items['name_3'] = LabelEncoder().fit_transform(items['name_3']).astype(np.int16)
items.drop(['item_name', 'name_1'], axis=1, inplace=True)
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
print('Use time:', time.time() - ts)
train['revenue'] = train['item_price'] *  train['item_cnt_day']

group = train.groupby(['date_block_num','shop_id','item_id']).agg({'item_cnt_day': ['sum']})

group.columns = ['item_cnt_month']
group.reset_index(inplace=True)
matrix = pd.merge(matrix, group, on=cols, how='left')
matrix['item_cnt_month'] = (matrix['item_cnt_month']
                                .fillna(0)
                                .clip(0,20)
                                .astype(np.float16))
test['date_block_num'] = 34
test['date_block_num'] = test['date_block_num'].astype(np.int8)
test['shop_id'] = test['shop_id'].astype(np.int8)
test['item_id'] = test['item_id'].astype(np.int16)

matrix = pd.concat([matrix, test], ignore_index=True, sort=False, keys=cols)
matrix.fillna(0, inplace=True) # 34 month
matrix = pd.merge(matrix, shops, on=['shop_id'], how='left')
matrix = pd.merge(matrix, items, on=['item_id'], how='left')
matrix = pd.merge(matrix, cats, on=['item_category_id'], how='left')
matrix['shop_city'] = matrix['shop_city'].astype(np.int8)
matrix['shop_category'] = matrix['shop_category'].astype(np.int8)
matrix['item_category_id'] = matrix['item_category_id'].astype(np.int8)
matrix['type_code'] = matrix['type_code'].astype(np.int8)
matrix['subtype_code'] = matrix['subtype_code'].astype(np.int8)
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id','item_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id','item_id', col+'-lag'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id','item_id'], how='left')
    return df
matrix = lag_feature(matrix, [1,2,3], 'item_cnt_month')
dict_simple = {'date_block_num': 'date', 'item_id': 'item', 'shop_id': 'shop', 
               'item_category_id': 'itemcate', 'item_price':'price', 
               
               'item_cnt_month': 'cnt', }

def sum_names(name_list):
    names = ''
    for x in name_list:
        names += x+'+'
    return names
    
def group_agg(matrix, groupby_feats, transform_feat, aggtype='mean'):
    group = matrix.groupby(groupby_feats).agg({transform_feat: [aggtype]})
    groupby_feats_simple = [dict_simple[x] if x in dict_simple.keys() else x
                            for x in groupby_feats]
    transform_feat_simple = dict_simple[transform_feat] \
                            if transform_feat in dict_simple.keys() else transform_feat
    group_name = f'{sum_names(groupby_feats_simple)[:-1]}-{aggtype.upper()}-{transform_feat_simple}'
    group.columns = [ group_name ]
    group.reset_index(inplace=True)
    return group, group_name
    
def add_groupmean_lag(matrix, groupby_feats, transform_feat, lags):
    group, group_name = group_agg(matrix, groupby_feats, transform_feat)
    
    matrix = pd.merge(matrix, group, on=groupby_feats, how='left')
    matrix[group_name] = matrix[group_name].astype(np.float16)
    if lags != []:
        matrix = lag_feature(matrix, lags, group_name)
        matrix.drop([group_name], axis=1, inplace=True)
    return matrix
transform_feat = 'item_cnt_month'


groupby_feats = ['date_block_num']
lags = [1]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'item_id']
lags = [1,2,3]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'shop_id']
lags = [1,2,3]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'item_category_id']
lags = [1]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'shop_id', 'item_category_id']
lags = [1]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'shop_id', 'item_id']
lags = [1]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'shop_id', 'subtype_code']
lags = [1]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'shop_city']
lags = [1]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)

groupby_feats = ['date_block_num', 'item_id', 'shop_city']
lags = [1]
matrix = add_groupmean_lag(matrix, groupby_feats, transform_feat, lags)
fetures_to_drop = []

transform_feat = 'item_price'
groupby_feats = ['item_id']
group, mean_price_col = group_agg(train, groupby_feats, 
                                  transform_feat, aggtype='mean')
matrix = pd.merge(matrix, group, on=groupby_feats, how='left')
matrix[mean_price_col] = matrix[mean_price_col].astype(np.float16)

transform_feat = 'item_price'
groupby_feats = ['date_block_num','item_id']
group, mean_monthlyprice_col = group_agg(train, groupby_feats, 
                                         transform_feat, aggtype='mean')
matrix = pd.merge(matrix, group, on=groupby_feats, how='left')
matrix[mean_monthlyprice_col] = matrix[mean_monthlyprice_col].astype(np.float16)

lags = [1,2,3]
matrix = lag_feature(matrix, lags, mean_monthlyprice_col)

for i in lags:
    matrix['delta_price-lag'+str(i)] = \
    (matrix[f'{mean_monthlyprice_col}-lag'+str(i)] - matrix[mean_price_col])\
    / matrix[mean_price_col]

matrix['delta_price-lag']=0
bool_loc = np.ones(len(matrix))==1
for i in lags:   
    matrix.loc[bool_loc, 'delta_price-lag'] = matrix.loc[bool_loc,'delta_price-lag'+str(i)]
    bool_loc &= matrix['delta_price-lag'+str(i)]==0
matrix['delta_price-lag'] = matrix['delta_price-lag'].astype(np.float16)
matrix['delta_price-lag'].fillna(0, inplace=True)

## Only keep 'delta_price_lag' feature
fetures_to_drop.append(mean_price_col)
fetures_to_drop.append(mean_monthlyprice_col)
for i in lags:
    fetures_to_drop += [f'{mean_monthlyprice_col}-lag'+str(i)]
    fetures_to_drop += ['delta_price-lag'+str(i)]

matrix.drop(fetures_to_drop, axis=1, inplace=True)
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
total_block_num = 35
date_block_num = np.arange(total_block_num)
date_block = [pd.Timestamp(2013, 1, 1)+pd.DateOffset(months=x) for x in date_block_num]


df_date = pd.DataFrame(date_block_num, columns=['date_block_num'])
df_date['date_block'] = date_block
df_date['year'] = df_date['date_block'].dt.year
df_date['month'] = df_date['date_block'].dt.month

for i in range(len(df_date)):
    day_to_count = 0
    calendar_matrix = calendar.monthcalendar(df_date['year'].iloc[i],df_date['month'].iloc[i])
    for j in range(7): # 7 days a week
        num_days = sum(1 for x in calendar_matrix if x[j] != 0)
        df_date.loc[i, f'week{j}'] = num_days
df_date = df_date[['date_block_num', 'year','month','week0','week1',
                   'week2','week3','week4','week5','week6']]  
df_date['days'] = df_date[['week0','week1','week2','week3','week4','week5','week6']].sum(axis=1)
df_date['year'] = df_date['year']-2012
df_date = df_date.astype(np.int8)

matrix = pd.merge(matrix, df_date, on=['date_block_num'], how='left')
matrix['item_shop_first_sale'] = \
matrix['date_block_num'] - matrix.groupby(['item_id','shop_id'])['date_block_num'].transform('min')

matrix['item_first_sale'] = \
matrix['date_block_num'] - matrix.groupby('item_id')['date_block_num'].transform('min')
matrix = matrix[matrix.date_block_num > 3]
def fill_na(df):
    for col in df.columns:
        if ('-lag' in col) & (df[col].isnull().any()):
            print(col)
            if ('cnt' in col):
                df[col].fillna(0, inplace=True)         
    return df

matrix = fill_na(matrix)
import pickle
import gc

del group
del items
del shops
del cats
del train

gc.collect();

matrix.to_pickle('../working/data.pkl')

del matrix
gc.collect();
data = pd.read_pickle('../working/data.pkl')

test  = pd.read_csv('../input/competitive-data-science-predict-future-sales/test.csv').set_index('ID')

print(len(data.columns))
data.columns
X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
Y_train = data[data.date_block_num < 33]['item_cnt_month']

X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
Y_valid = data[data.date_block_num == 33]['item_cnt_month']
X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)

del data
gc.collect();
X_train.info()
lgb_train = lgb.Dataset(X_train, Y_train)
lgb_eval = lgb.Dataset(X_valid, Y_valid, reference=lgb_train)

del X_train
gc.collect();
def rmsle(y, y_pred):
    return np.sqrt(mean_squared_error(y, y_pred))

params = {'num_leaves': 2000, 'max_depth': 19, 'max_bin': 107, 'n_estimators': 3747,
          'bagging_freq': 1, 'bagging_fraction': 0.7135681370918421, 
          'feature_fraction': 0.49446461478601994, 'min_data_in_leaf': 88, 
          'learning_rate': 0.015980721586917768, 'num_threads': 3, 
          'min_sum_hessian_in_leaf': 6,
         
          'random_state' : RANDOM_SEED,
          'verbosity' : 1,
          'bagging_seed' : RANDOM_SEED,
          'boost_from_average' : 'true',
          'boost' : 'gbdt',
          'metric' : 'rmse',}

model = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=[lgb_train,lgb_eval],
                early_stopping_rounds=20,
                verbose_eval=1,
                )
y_pred = model.predict(X_valid)
rmsle(Y_valid, y_pred)
Y_pred = model.predict(X_valid).clip(0, 20)
Y_test = model.predict(X_test).clip(0, 20)

submission = pd.DataFrame({
    "ID": test.index, 
    "item_cnt_month": Y_test
})
submission.to_csv('AbdN_submission.csv', index=False)

pickle.dump(Y_pred, open('lgb_train.pickle', 'wb'))
pickle.dump(Y_test, open('lgb_test.pickle', 'wb'))
from lightgbm import plot_importance

def plot_features(booster, figsize):    
    fig, ax = plt.subplots(1,1,figsize=figsize)
    return plot_importance(booster=booster, ax=ax)

plot_features(model, (10,14))
feat_importance = model.feature_importance()
df_importance = pd.DataFrame(feat_importance, columns=['importance'], index=X_test.columns)
df_importance = df_importance.sort_values(by='importance', ascending=False)
df_importance.index
df_importance