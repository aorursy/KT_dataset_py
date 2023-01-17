import warnings

warnings.filterwarnings('ignore')

import pandas as pd

import numpy as np

import dask.dataframe as dd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)

import matplotlib.pyplot as plt

import seaborn as sns; sns.set()

import lightgbm as lgb

#import dask_xgboost as xgb

#import dask.dataframe as dd

from sklearn import preprocessing, metrics

from sklearn.preprocessing import LabelEncoder

import gc

import os

from sklearn.cluster import KMeans

from tqdm import tqdm

from scipy.sparse import csr_matrix

from datetime import timedelta



for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns: #columns毎に処理

        col_type = df[col].dtypes

        if col_type in numerics: #numericsのデータ型の範囲内のときに処理を実行. データの最大最小値を元にデータ型を効率的なものに変更

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2

    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))

    return df



# カテゴリ変数化（NaNはそのままになる）

def encode_categorical(df, cols):

    

    for col in cols:

        # Leave NaN as it is.

        le = LabelEncoder()

        not_null = df[col][df[col].notnull()]

        df[col] = pd.Series(le.fit_transform(not_null), index=not_null.index)



    return df
import IPython



def display(*dfs, head=True):

    for df in dfs:

        IPython.display.display(df.head() if head else df)
h = 28 

max_lags = 366

tr_last = 1913

fday = pd.to_datetime("2016-04-25") 
# get data

def create_dt(is_train = True, nrows = 10**100):

    

    cal = pd.read_csv('../input/m5-forecasting-accuracy/calendar.csv')

    cal['date'] = pd.to_datetime(cal['date'])

    cal['is_weekend'] = 1

    cal['is_weekend'] = cal['is_weekend'].where(cal['weekday'].isin(['Saturday', 'Sunday']), 0)

    cal = reduce_mem_usage(cal)

    

    prices = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')

    prices = reduce_mem_usage(prices)

    

    if is_train:

        dt = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv', nrows=nrows)

    else:

        dt = pd.read_csv('../input/m5-forecasting-accuracy/sales_train_validation.csv', nrows=nrows)

        dt = dt.drop(['d_' + str(i) for i in range(1, tr_last - max_lags + 1)], axis = 1)

        

        na_cols = list(['d_' + str(i) for i in range(tr_last + 1, tr_last + 2 * h + 1)])

        for i in na_cols:

            dt[i] = np.nan

            

    dt = dt[dt['store_id'] == 'CA_1']

            

    dt = pd.melt(dt, id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'], var_name = 'd', value_name = 'sales')

    dt = pd.merge(dt, cal[['d', 'date', 'is_weekend', 'wm_yr_wk', 'event_name_1', 'snap_CA', 'snap_TX', 'snap_WI']], on = 'd', how = 'left')

    dt = pd.merge(dt, prices, on = ['store_id', 'item_id', 'wm_yr_wk'], how = 'left')

    

    dt = reduce_mem_usage(dt)

    

    return dt
# make feature

def create_fea(dt):

    

    for lag in [7, 28, 29]:

#     for lag in [28, 29]:

        dt[f"lag_{lag}"] = dt.groupby(["id"])["sales"].transform(lambda x: x.shift(lag))

        

    # min_periods = 1 とすることで R の結果と一致

    for win in [7, 30, 90, 180]:

        dt[f"roll_mean_28_{win}"] = dt.groupby(["id"])["sales"].transform(lambda x: x.shift(28).rolling(win, min_periods = 1).mean())

    

    # min_periods を指定しないことで R の結果と一致

    for win in [28]:

        dt[f"roll_max_28_{win}"] = dt.groupby(["id"])["sales"].transform(lambda x: x.shift(28).rolling(win).max())

        dt[f"roll_var_28_{win}"] = dt.groupby(["id"])["sales"].transform(lambda x: x.shift(28).rolling(win).var())

        

    # R と計算結果（fillna(0）したあとの平均）が微妙に異なる（astype(float)のせい？）

    # dt['price_change_1'].astype(float).fillna(0).mean()

    # dt["shift_price_t1"].astype(float).fillna(0).mean()

    # dt['sell_price'].astype(float).fillna(0).mean()

    dt["shift_price_t1"] = dt.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1))

    dt['price_change_1'] = dt['sell_price'] / dt['shift_price_t1'] -1

    dt = dt.drop(['shift_price_t1'], axis = 1)

    

    # R と計算結果（fillna(0）したあとの平均）が微妙に異なる（astype(float)のせい？）

    dt["rolling_price_max_t365"] = dt.groupby(["id"])["sell_price"].transform(lambda x: x.shift(1).rolling(365).max())

    dt["price_change_365"] = dt["sell_price"] / dt["rolling_price_max_t365"] - 1

    dt = dt.drop(['rolling_price_max_t365'], axis = 1)

    

    # event_name_1 はNaN（文字列）埋めしておく

    dt['event_name_1'] = dt['event_name_1'].fillna('NaN')

    # カテゴリ変数化

    dt = encode_categorical(dt, ["item_id", "state_id", "dept_id", "cat_id", "event_name_1"])

    

    dt['wday'] = dt['date'].dt.weekday

    dt['mday'] = dt['date'].dt.day

    dt['week'] = dt['date'].dt.week

    dt['month'] = dt['date'].dt.month

    dt['quarter'] = dt['date'].dt.quarter

    dt['year'] = dt['date'].dt.year

    # 不要な列

#     dt['store_id'] = np.nan

#     dt['d'] = np.nan

#     dt['wm_yr_wk'] = np.nan

    dt = dt.drop(['store_id', 'd', 'wm_yr_wk'], axis = 1)

    

    dt = reduce_mem_usage(dt)

    

    return dt
tr = create_dt()

tr = create_fea(tr)

gc.collect()
features = [

    'id', 'item_id', 'dept_id', 'cat_id', 'state_id', 'sales', 'date', 'event_name_1', 'snap_CA', 'snap_TX', 'snap_WI',

    'sell_price', 

    'lag_7',

    'lag_28', 'lag_29',

    'roll_mean_28_7', 'roll_mean_28_30', 'roll_mean_28_90', 'roll_mean_28_180', 'roll_max_28_28', 'roll_var_28_28',

    'price_change_1', 'price_change_365', 'wday', 'mday', 'week', 'month', 'quarter', 'year', 'is_weekend']



tr = tr[features].dropna()

y = tr.sales

# indexの定義

idx = tr[tr['date'] <= tr['date'].max() - timedelta(days = h)].index



# 不要な列

tr['id'] = np.nan

tr['sales'] = np.nan

tr['date'] = np.nan



tr = tr.drop(['id', 'sales', 'date'], axis = 1)

gc.collect()
# カテゴリ変数への変換

cats = [

    "item_id", "state_id", "dept_id", "cat_id", "event_name_1", 

    "wday", "mday", "week", "month", "quarter", "year", "is_weekend", "snap_CA", "snap_TX", "snap_WI"]



def change_cat(dt):

    for i in cats:

        dt[i] = dt[i].astype('category')

    return dt



tr = change_cat(tr)
# tr.dtypes
# 型変換しておく

def change_type(dt):

    for col in dt.select_dtypes(include='float16').astype(float).columns:

        dt[col] = dt[col].astype(float)

    return dt

tr = change_type(tr)
# tr.describe()
# lgbm

xtr = lgb.Dataset(

    tr[tr.index.isin(idx)],

    y[y.index.isin(idx)],

    categorical_feature = cats

)

xval = lgb.Dataset(

    tr[~tr.index.isin(idx)],

    y[~y.index.isin(idx)],

    categorical_feature = cats

)
# lgbm

params = {

#     'boosting_type': 'gbdt',

    'metric': 'rmse',

    'objective': 'poisson',

    'force_row_wise': True,

#     'n_jobs': -1,

#     'seed': 236,

    'learning_rate': 0.075,

    'sub_feature': 0.8,

    'sub_row': 0.75,

    'bagging_fraction': 1,

    'lambda_l2': 0.1,

    'nthread': 4

#     'colsample_bytree': 0.75

}



model = lgb.train(

    params,

    xtr,

    num_boost_round = 2000,

#     num_boost_round = 100,

    early_stopping_rounds = 400,

#     early_stopping_rounds = 10,

    valid_sets = [xtr, xval],

    verbose_eval = 200)
# get test data

te = create_dt(False)
for day in pd.date_range(start = fday, end = fday + timedelta(days = 2 * h - 1), freq='D'):

    print(day)

    tst = te[(te.date >= day - timedelta(days = max_lags))&(te.date <= day)]

    tst = create_fea(tst)

    tst = change_cat(tst)

    tst = change_type(tst)

    tst = tst[tst.date == day].drop(['id', 'sales', 'date'], axis = 1)

    te.loc[te['date'] == day, 'sales'] = model.predict(tst)

    gc.collect()
submission = pd.read_csv('../input/m5-forecasting-accuracy/sample_submission.csv')
predictions = te[(te.date >= fday)&(te.date <= fday + timedelta(days = h -1))][['id', 'date', 'sales']]

predictions = pd.pivot(predictions, index = 'id', columns = 'date', values = 'sales').reset_index()

predictions.columns = ['id'] + ['F' + str(i + 1) for i in range(28)]



evaluation_rows = [row for row in submission['id'] if 'evaluation' in row] 

evaluation = submission[submission['id'].isin(evaluation_rows)]



validation = submission[['id']].merge(predictions, on = 'id')

final = pd.concat([validation, evaluation])

final.to_csv('submission.csv', index = False)
# importanceを表示する

importance = pd.DataFrame(model.feature_importance(), index=tr.columns, columns=['importance'])

importance.sort_values(by = 'importance', ascending = False)
te.groupby(['date'])['sales'].sum().plot()