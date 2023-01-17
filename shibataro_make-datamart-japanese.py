# system

import sys

import os

import pickle

from datetime import datetime



# data manipulation

import numpy as np

import pandas as pd



# gabarge collection

import gc



# path

print("CWD: " + os.getcwd())

# データの格納先

DATA = '/kaggle/working/narrowed_data'

if not os.path.exists(DATA):

    os.mkdir(DATA)
# メモリを節約する関数 (copied from https://www.kaggle.com/ratan123/m5-forecasting-lightgbm-with-timeseries-splits)

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage().sum() / 1024**2    

    for col in df.columns:

        col_type = df[col].dtypes

        if col_type in numerics:

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
# データ読み込み用関数 (modified from https://www.kaggle.com/ratan123/m5-forecasting-lightgbm-with-timeseries-splits)

# 2014/5/1以降のデータのみ用いる. 

# 5分ほどかかる

MIN_DATE = datetime(2014, 5, 1)



def read_data():

    print('Reading files...')

    # calendar.csv

    print('calendar.csv')

    calendar = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/calendar.csv')

    # 期間絞り込み

    calendar = calendar.loc[

        pd.to_datetime(calendar['date'], format='%Y/%m/%d') >= MIN_DATE

    ]

    # メモリ節約

    calendar = calendar.drop(['weekday', 'wday', 'month', 'year', 'event_name_2', 'event_type_2'], axis=1)

    calendar = reduce_mem_usage(calendar)

    print()

    

    # 後にsell_price.csvの期間を絞るために2014/1/1に対応する"wm_yr_wk"を格納

    MIN_WM_YR_WK = calendar['wm_yr_wk'].min()

    # 後にsales_train_evaluation.csvの期間を絞るために2014/5/1以降の"d_xxx"を格納

    day_cols = calendar['d'].values.tolist()

    

    # sell_prices.csv

    print('sell_prices.csv')

    sell_prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')

    # 期間の絞り込み

    sell_prices = sell_prices.loc[sell_prices['wm_yr_wk'] >= MIN_WM_YR_WK]

    # メモリ節約

    sell_prices = reduce_mem_usage(sell_prices)

    print()

    

    # sales_train_evaluation.csv

    print('sales_train_evaluation.csv')

    sales_train_evaluation = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sales_train_evaluation.csv')

    # メモリ節約

    sales_train_evaluation = reduce_mem_usage(sales_train_evaluation)

    # 直近2年に絞り込み

    cols = list(sales_train_evaluation.columns)

    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

    use_cols = id_cols + day_cols

    sales_train_evaluation.drop(

        [col for col in cols if col not in use_cols],

        axis=1,

        inplace=True

    )

    print()

    

    # sample_submission.csv

    print('sample_submission.csv')

    submission = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sample_submission.csv')

    # メモリ節約

    submission = reduce_mem_usage(submission)

    print()

    

    return calendar, sell_prices, sales_train_evaluation, submission



calendar, sell_prices, sales, submission = read_data()

gc.collect()

print('data loading is done')
# submissionデータの"day"のマッピング用dict

day_map_dict_val = {'F'+str(i): 'd_'+str(1913+i) for i in range(1, 29)}

day_map_dict_eva = {'F'+str(i): 'd_'+str(1941+i) for i in range(1, 29)}



# 後からも使えるようにマッピング用dictをpickleで保存しておく

with open(DATA+'/day_map_dict_val.pkl', 'wb') as f:

    pickle.dump(day_map_dict_val, f)

with open(DATA+'/day_map_dict_eva.pkl', 'wb') as f:

    pickle.dump(day_map_dict_eva, f)



# データの横持ち->縦持ち変換

def melt_df(sales, submission):

    # wide -> long

    sales = pd.melt(

        sales,

        id_vars=['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id'],

        var_name = 'day',

        value_name = 'demand'

    )



    submission = pd.melt(

        submission,

        id_vars=['id'],

        var_name='day',

        value_name='demand'

    )

    

    # submissionの"day"をmappingしたカラム"day"を作成

    submission['day'].update(

        submission.loc[submission['id'].str.contains('validation'), 'day'].map(day_map_dict_val)

    )

    submission['day'].update(

        submission.loc[submission['id'].str.contains('evaluation'), 'day'].map(day_map_dict_eva)

    )

    

    # submissionについて'item_id'と'store_id'を正規表現で抽出

    submission['item_id'] = submission.loc[:, 'id'].str.extract('(.+[0-9]{3})')

    submission['store_id'] = submission.loc[:, 'id'].str.extract('(.?[CA|TX|WI]_[0-9])')

    

    return sales, submission



sales, submission = melt_df(sales, submission)
calendar.head()
calendar.tail()
sell_prices.head()
sell_prices.tail()
sales.head()
sales.tail()
submission.head()
submission.tail()
def merge_df(df, calendar, sell_price):

    merged_dm = pd.merge(

        df, calendar,

        left_on='day', right_on='d',

        how='left',

        copy=False

    ).drop('d', axis=1)

    

    del df, calendar

    

    merged_dm = pd.merge(

        merged_dm, sell_price,

        on=['store_id', 'item_id', 'wm_yr_wk'],

        how='left',

        copy=False

    )

    

    del sell_price

    gc.collect()

    

    return merged_dm
sales_data = merge_df(

    sales,

    calendar,

    sell_prices

)
sales_data.head()
sales_data.tail()
with open(DATA+'/sales_data.pkl', 'wb') as f:

    pickle.dump(sales_data, f)
del sales

gc.collect()
submission_data = merge_df(

    submission,

    calendar,

    sell_prices

)
submission_data.head()
submission_data.tail()
with open(DATA+'/submission_data.pkl', 'wb') as f:

    pickle.dump(submission_data, f)
del submission, calendar, sell_prices

gc.collect()
%reset
!ls /kaggle/input/m5-narrowed-data/narrowed_data
# system

import sys

import os

import pickle

from datetime import datetime



# data manipulation

import numpy as np

import pandas as pd

pd.set_option('display.max_columns', 500)

pd.set_option('display.max_rows', 500)



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

sns.set()



# ML model

import lightgbm as lgb

# optunaでハイパラチューニングを行う場合

#import optuna.integration.lightgbm as lgb



# gabarge collection

import gc



# path

print("CWD: " + os.getcwd())

DATA = '/kaggle/input/m5-narrowed-data/narrowed_data'
# load data

# 一旦学習データだけを読み込む

with open(DATA+'/sales_data.pkl', 'rb') as f:

    sales_data = pickle.load(f)
sales_data.head()
# 実行時間短縮のために期間の絞り込み

MIN_DATE = datetime(2015, 5, 1)

sales_data = sales_data.loc[

    pd.to_datetime(sales_date['date'], format='%Y-%m-%d') >= MIN_DATE

]

sales_data.head()