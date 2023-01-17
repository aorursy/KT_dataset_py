import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import product
from xgboost import XGBRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import calendar
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sales = pd.read_csv('../input/assignment/Assignment.csv')
sales.head()
sales['item_price']=sales['Invoice Value']/sales['Quantity']
sales.head()
sales=sales[["Customer ID","Quantity","Invoice Value","Date","item_price"]]
sales.columns = ["shop_id","item_cnt_day","Invoice Value","Date","item_price"]
sales = sales.sort_values(['Date'])
sales['month'] = pd.DatetimeIndex(sales['Date']).month
sales['year'] = pd.DatetimeIndex(sales['Date']).year
def f(row):
    if row['year'] == 2018:
        val = row['month']-1
    else:
        val = row['month']+12-1
    return val

sales['date_block_num'] = sales.apply(f, axis=1)
sales.head()
sns.boxplot(x=sales.item_cnt_day)
sns.boxplot(x=sales.item_price)
train = sales[(sales.item_price < 80) & (sales.item_price > 0)]
train = train[sales.item_cnt_day < 600]
index_cols = ['shop_id', 'date_block_num']

df = [] 
for block_num in train['date_block_num'].unique():
    cur_shops = train.loc[sales['date_block_num'] == block_num, 'shop_id'].unique()
    df.append(np.array(list(product(*[cur_shops, [block_num]])),dtype='int32'))

index_cols
df = pd.DataFrame(np.vstack(df), columns = index_cols,dtype=np.int32)
#Add month sales
group = train.groupby(['date_block_num','shop_id']).agg({'item_cnt_day': ['sum']})
group.columns = ['item_cnt_month']
group.reset_index(inplace=True)

df = pd.merge(df, group, on=index_cols, how='left')
df['item_cnt_month'] = (df['item_cnt_month']
                                .fillna(0)
                                .clip(0,20)
                                .astype(np.float16))
df.head(5)
df
sales.head()
train=sales[(sales.item_cnt_day >= 1) & (sales.Date >='2018-01-01') & (sales.Date <='2019-3-31')]
train.shape
test=sales[(sales.item_cnt_day >= 1) & (sales.Date >='2019-01-01') & (sales.Date <='2019-3-31')]
test.shape
def lag_feature(df, lags, col):
    tmp = df[['date_block_num','shop_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id', col+'_lag_'+str(i)]
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id'], how='left')
        df[col+'_lag_'+str(i)] = df[col+'_lag_'+str(i)].astype('float16')
    return df
#Add sales lags for last 3 months
df = lag_feature(df, [1, 2, 3], 'item_cnt_month')
df
train
#Add avg shop/item price

index_cols = ['shop_id', 'date_block_num']
group = train.groupby(index_cols)['item_price'].mean().reset_index().rename(columns={"item_price": "avg_shop_price"}, errors="raise")
df = pd.merge(df, group, on=index_cols, how='left')

df['avg_shop_price'] = (df['avg_shop_price']
                                .fillna(0)
                                .astype(np.float16))


df
#Add target encoding for item/shop for last 3 months 
item_id_target_mean = df.groupby(['date_block_num', 'shop_id'])['item_cnt_month'].mean().reset_index().rename(columns={
    "item_cnt_month": "item_shop_target_enc"}, errors="raise")

df = pd.merge(df, item_id_target_mean, on=['date_block_num', 'shop_id'], how='left')

df['item_shop_target_enc'] = (df['item_shop_target_enc']
                                .fillna(0)
                                .astype(np.float16))

df = lag_feature(df, [1, 2, 3], 'item_shop_target_enc')
df.drop(['item_shop_target_enc'], axis=1, inplace=True)
df
def lag_feature_adv(df, lags, col):
    tmp = df[['date_block_num','shop_id',col]]
    for i in lags:
        shifted = tmp.copy()
        shifted.columns = ['date_block_num','shop_id', col+'_lag_'+str(i)+'_adv']
        shifted['date_block_num'] += i
        df = pd.merge(df, shifted, on=['date_block_num','shop_id'], how='left')
        df[col+'_lag_'+str(i)+'_adv'] = df[col+'_lag_'+str(i)+'_adv'].astype('float16')
    return df

df = lag_feature_adv(df, [1, 2, 3], 'item_cnt_month')
df
df.fillna(0, inplace=True)
df = df[(df['date_block_num'] > 2)]
df.head()
df.tail()
df.columns
#Save dataset
df.drop(['ID'], axis=1, inplace=True, errors='ignore')
df.to_pickle('df.pkl')
df = pd.read_pickle('df.pkl')
df.info()
X_train = df[df.date_block_num < 11].drop(['item_cnt_month'], axis=1)
Y_train = df[df.date_block_num < 11]['item_cnt_month']
X_valid = df[df.date_block_num == 11].drop(['item_cnt_month'], axis=1)
Y_valid = df[df.date_block_num == 11]['item_cnt_month']
X_test = df[df.date_block_num ==12].drop(['item_cnt_month'], axis=1)
del df
X_test.tail()
feature_name = X_train.columns.tolist()
feature_name
feature_name = X_train.columns.tolist()

params = {
    'objective': 'mse',
    'metric': 'rmse',
    'num_leaves': 2 ** 7 - 1,
    'learning_rate': 0.005,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'seed': 1,
    'verbose': 1
}

feature_name_indexes = [ 
                        'country_part', 
                        'item_category_common',
                        'item_category_code', 
                        'city_code',
]

lgb_train = lgb.Dataset(X_train[feature_name], Y_train)
lgb_eval = lgb.Dataset(X_valid[feature_name], Y_valid, reference=lgb_train)

evals_result = {}
gbm = lgb.train(
        params, 
        lgb_train,
        num_boost_round=3000,
        valid_sets=(lgb_train, lgb_eval), 
        feature_name = feature_name,
        verbose_eval=5, 
        evals_result = evals_result,
        early_stopping_rounds = 100)

lgb.plot_importance(
    gbm, 
    max_num_features=50, 
    importance_type='gain', 
    figsize=(12,8));
feature_name
test.head()
X_test
Y_test = gbm.predict(X_test[feature_name]).clip(0, 20)

submission = pd.DataFrame({
    "ID": X_test.shop_id, 
    "item_cnt_month": Y_test
})
submission.to_csv('gbm_submission.csv', index=False)
submission
