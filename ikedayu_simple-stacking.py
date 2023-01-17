%matplotlib inline
from IPython.display import Image

import os, sys, re, datetime, gc
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

import lightgbm as lgb
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn import feature_extraction
from sklearn import preprocessing
from tqdm import tqdm_notebook

pd.set_option('display.max_rows', 600)
pd.set_option('display.max_columns', 50)

for p in [np, pd, sklearn, lgb]:
    print (p.__name__, p.__version__)
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
def evaluate(true, pred):
    return np.sqrt(mean_squared_error(true.clip(0.,20.), pred.clip(0.,20.)))
random_seed = 1021
lgb_params = {
    'feature_fraction': 0.75,
    'metric': 'rmse',
    'nthread': -1,
    'min_data_in_leaf': 2**7,
    'bagging_fraction': 0.75,
    'learning_rate': 0.03,
    'objective': 'mse',
    'num_leaves': 2**7,
    'bagging_freq': 1,
    'verbose': 0
}

xgb_params = {
    'eta': 0.2, 
    'max_depth': 4, 
    'objective': 'reg:linear', 
    'eval_metric': 'rmse', 
    'seed': random_seed, 
    'silent': True
}

cat_params = {
    'iterations': 100, 
    'learning_rate': 0.2, 
    'depth': 7, 
    'loss_function': 'RMSE', 
    'eval_metric': 'RMSE', 
    'random_seed': random_seed, 
    'od_type': 'Iter', 
    'od_wait': 20
}
data_dir = Path('../input')
# data_dir = Path.home()/'.kaggle/competitions/competitive-data-science-predict-future-sales'
train = pd.read_csv(data_dir/'sales_train.csv')
test = pd.read_csv(data_dir/'test.csv')
items = pd.read_csv(data_dir/'items.csv')
item_cats = pd.read_csv(data_dir/'item_categories.csv')
shops = pd.read_csv(data_dir/'shops.csv')
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
items['item_name_len'] = items['item_name'].map(len) #Lenth of Item Description
items['item_name_wc'] = items['item_name'].map(lambda x: len(str(x).split(' '))) #Item Description Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(items['item_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    items['item_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
items.head()
#Text Features
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
item_cats['item_category_name_len'] = item_cats['item_category_name'].map(len)  #Lenth of Item Category Description
item_cats['item_category_name_wc'] = item_cats['item_category_name'].map(lambda x: len(str(x).split(' '))) #Item Category Description Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(item_cats['item_category_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    item_cats['item_category_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
item_cats.head()
#Text Features
feature_cnt = 25
tfidf = feature_extraction.text.TfidfVectorizer(max_features=feature_cnt)
shops['shop_name_len'] = shops['shop_name'].map(len)  #Lenth of Shop Name
shops['shop_name_wc'] = shops['shop_name'].map(lambda x: len(str(x).split(' '))) #Shop Name Word Count
txtFeatures = pd.DataFrame(tfidf.fit_transform(shops['shop_name']).toarray())
cols = txtFeatures.columns
for i in range(feature_cnt):
    shops['shop_name_tfidf_' + str(i)] = txtFeatures[cols[i]]
shops.head()
#Make Monthly
train['date'] = pd.to_datetime(train['date'], format='%d.%m.%Y')
train['month'] = train['date'].dt.month
train['year'] = train['date'].dt.year
train = train.drop(['date','item_price'], axis=1)
train = train.groupby([c for c in train.columns if c not in ['item_cnt_day']], as_index=False)[['item_cnt_day']].sum()
train = train.rename(columns={'item_cnt_day':'item_cnt_month'})

#Monthly Mean
shop_item_monthly_mean = train[['shop_id','item_id','item_cnt_month']].groupby(['shop_id','item_id'], as_index=False)[['item_cnt_month']].mean()
shop_item_monthly_mean = shop_item_monthly_mean.rename(columns={'item_cnt_month':'item_cnt_month_mean'})

#Add Mean Feature
train = pd.merge(train, shop_item_monthly_mean, how='left', on=['shop_id','item_id'])

#Last Month (Oct 2015)
shop_item_prev_month = train[train['date_block_num']==33][['shop_id','item_id','item_cnt_month']]
shop_item_prev_month = shop_item_prev_month.rename(columns={'item_cnt_month':'item_cnt_prev_month'})
shop_item_prev_month.head()

#Add Previous Month Feature
train = pd.merge(train, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)

#Items features
train = pd.merge(train, items, how='left', on='item_id')

#Item Category features
train = pd.merge(train, item_cats, how='left', on='item_category_id')

#Shops features
train = pd.merge(train, shops, how='left', on='shop_id')
train.head()
test['month'] = 11
test['year'] = 2015
test['date_block_num'] = 34

#Add Mean Feature
test = pd.merge(test, shop_item_monthly_mean, how='left', on=['shop_id','item_id']).fillna(0.)

#Add Previous Month Feature
test = pd.merge(test, shop_item_prev_month, how='left', on=['shop_id','item_id']).fillna(0.)

#Items features
test = pd.merge(test, items, how='left', on='item_id')

#Item Category features
test = pd.merge(test, item_cats, how='left', on='item_category_id')

#Shops features
test = pd.merge(test, shops, how='left', on='shop_id')
test['item_cnt_month'] = 0.
test.head()
sales = pd.concat([train, test.drop(columns='ID')], axis=0, sort=False)
# limit data
# sales = sales[sales['shop_id'].isin([26, 27, 28])]
for c in ['shop_name','item_name','item_category_name']:
    lbl = preprocessing.LabelEncoder()
    lbl.fit(sales[c].unique())
    sales[c] = lbl.transform(sales[c].astype(str))
all_block_num = sales['date_block_num'].unique()
train_block_num = train['date_block_num'].unique()
all_data = sales
index_cols = ['shop_id', 'item_id', 'date_block_num']
cols_to_rename = list(all_data.columns.difference(index_cols)) 

shift_range = [1, 2, 3, 4, 5, 12]

origin_cols = [c for c in all_data.columns if 'tfidf' not in c]

for month_shift in tqdm_notebook(shift_range):
    train_shift = all_data[origin_cols].copy()
    train_shift['date_block_num'] = train_shift['date_block_num'] + month_shift
    
    to_rename = lambda x: f'{x}_lag_{month_shift}' if x in cols_to_rename else x
    train_shift = train_shift.rename(columns=to_rename)
    
    all_data = pd.merge(all_data, train_shift, on=index_cols, how='left').fillna(0)

del train_shift

all_data = all_data.query('date_block_num >= 12')
all_data.rename(columns={'item_cnt_month': 'target'}, inplace=True)
all_data['target'] = np.log(all_data['target'].clip(0.,20.))
submit_data = all_data.query('date_block_num == 34')
all_data = all_data.query('date_block_num < 34')
all_data = downcast_dtypes(all_data)
dates = all_data['date_block_num']
fit_cols = [col for col in all_data.columns if col[-1] in [str(item) for item in shift_range]] 
to_drop_cols = list(set(list(all_data.columns)) - (set(fit_cols)|set(index_cols))) + ['date_block_num'] 

last_block = dates.max()

dates_train = dates[dates < last_block]
dates_test = dates[dates == last_block]

X_train = all_data.query('date_block_num < @last_block').drop(to_drop_cols, axis=1)
X_test = all_data.query('date_block_num == @last_block').drop(to_drop_cols, axis=1)

y_train = all_data.query('date_block_num < @last_block')['target']
y_test = all_data.query('date_block_num == @last_block')['target']

X_submit = submit_data.drop(to_drop_cols, axis=1)
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
models = {
    "Ridge": Ridge(), 
    "RF": RandomForestRegressor(n_jobs=-1), 
    "XGB": XGBRegressor(n_jobs=-1, **xgb_params), 
    "CB": CatBoostRegressor(**cat_params), 
    "LGB": LGBMRegressor(n_jobs=-1, **lgb_params)
}
base_preds = []
base_preds_submit = []

for name, model in tqdm_notebook(models.items()):
    model.fit(X_train.values, y_train.values)
    base_pred = model.predict(X_test.values)
    rmse = evaluate(y_test, base_pred)
    print(f'{name} RMSE: ', rmse)
    base_preds.append(base_pred)
    
    base_preds_submit.append(model.predict(X_submit.values))
X_test_level2 = np.column_stack(base_preds)
X_submit_level2 = np.column_stack(base_preds_submit)
range_train_level2 = [27, 28, 29, 30, 31, 32]

dates_train_level2 = dates_train[dates_train.isin(range_train_level2)]
y_train_level2 = y_train[dates_train.isin(dates_train_level2)]
X_train_level2 = np.zeros([y_train_level2.shape[0], len(models)])

for current_block_num in tqdm_notebook(range_train_level2):
    X_train = all_data.query('date_block_num < @current_block_num').drop(to_drop_cols, axis=1)
    X_test = all_data.query('date_block_num == @current_block_num').drop(to_drop_cols, axis=1)
    
    y_train = all_data.query('date_block_num < @current_block_num')['target']
    n_test = y_test.shape[0]

    preds = []
    for name, model in tqdm_notebook(models.items()):
        model.fit(X_train.values, y_train.values)
        pred = model.predict(X_test.values)
        preds.append(pred)
    
    idxs = dates_train_level2.reset_index().index[dates_train_level2 == current_block_num]
    X_train_level2[idxs] = np.column_stack(preds)
X_train_level2.shape
meta_model = LinearRegression()
meta_model.fit(X_train_level2, y_train_level2)
train_preds = meta_model.predict(X_train_level2)
train_rmse = evaluate(y_train_level2, train_preds)
print('Stacking Train RMSE: ', train_rmse)

test_preds = meta_model.predict(X_test_level2)
test_rmse = evaluate(y_test, test_preds)
print('Stacking Test RMSE: ', test_rmse)
submit_pred = meta_model.predict(X_submit_level2)
test['item_cnt_month'] = submit_pred
test[['ID', 'item_cnt_month']].to_csv('stacking_20181021_v8.csv', index=False)
test[['ID', 'item_cnt_month']]

