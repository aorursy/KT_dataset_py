import os

import numpy as np

import pandas as pd

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns



pd.set_option('max_colwidth', 500)

pd.set_option('max_columns', 500)

pd.set_option('max_rows', 100)



import gc

from sklearn.metrics import mean_squared_error



DATA = '../input/ashrae-energy-prediction/'



print(os.listdir(DATA))
%load_ext autoreload

%autoreload 2
def reduce_mem_usage(df, force_obj_in_category=True, debug=True):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage. 

        This function originates from https://www.kaggle.com/gemartin/load-data-reduce-memory-usage

    """

    if debug:

        start_mem = df.memory_usage(deep=True).sum() / 1024**2

        print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        col_type = df[col].dtype

        

        if col_type != object and df[col].dtype.name != 'category' and 'datetime' not in col_type.name:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':

                for i_type in [np.int8, np.int16, np.int32, np.int64]:

                    if c_min > np.iinfo(i_type).min and c_max < np.iinfo(i_type).max:

                        df[col] = df[col].astype(i_type)

                        break

            elif str(col_type)[:4] == 'uint':

                for i_type in [np.uint8, np.uint16, np.uint32, np.uint64]:

                    if c_max < np.iinfo(i_type).max:

                        df[col] = df[col].astype(i_type)

                        break

            elif col_type == bool:

                df[col] = df[col].astype(np.uint8)

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        elif force_obj_in_category and 'datetime' not in col_type.name:

            df[col] = df[col].astype('category')



    if debug:

        end_mem = df.memory_usage(deep=True).sum() / 1024**2

        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
def get_data(v):

    '''

    Read input data and merge with building and weather information.

    

    Parameters

    -----------

    v: string [train/test]

        Read either tarining or test dataset

    '''

    df = pd.read_csv(f'{DATA}{v}.csv')#, parse_dates=['timestamp'])

    df = reduce_mem_usage(df)

    

#     df_buildings = pd.read_csv(f'{DATA}building_metadata.csv')

#     # drop arbitrary set of potentially irrelevant features

#     df_buildings.drop(['primary_use', 'floor_count'],

#                       axis=1, inplace=True)

#     df_buildings = reduce_mem_usage(df_buildings, debug=False)

#     df = df.merge(df_buildings, how='left', on='building_id')

    

#     df_weather = pd.read_csv(f'{DATA}weather_{v}.csv')

#     # drop arbitrary set of potentially irrelevant features

#     df_weather.drop(['cloud_coverage','dew_temperature','precip_depth_1_hr','sea_level_pressure','wind_direction','wind_speed'],

#                     axis=1, inplace=True)

#     df_weather = reduce_mem_usage(df_weather, debug=False)

#     df = df.merge(df_weather, how='left', on=['site_id', 'timestamp'])

    

    if 'meter_reading' in df.columns:

        df['target'] = np.log1p(df['meter_reading']).astype(np.float32)

        del df['meter_reading']

    

    print(f'Final memory size of {v} dataset = {df.memory_usage(deep=True).sum()/2**20:.2f} MB')

        

    return df
df_trn = get_data('train')
def get_ts_features(df):

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    df['ts_day_of_month'] = df['timestamp'].dt.day.astype(np.uint8)

    df['ts_month'] = df['timestamp'].dt.month.astype(np.uint8)

    df['ts_hour'] = df['timestamp'].dt.hour.astype(np.uint8)

    df['ts_day_of_week'] = (df['timestamp'].dt.weekday<=4).astype(np.uint8)

    del df['timestamp']

    return df
df_trn = get_ts_features(df_trn)
df_trn.head()
df_ts = {'d':[], 'm':[], 'h':[], 'w':[]}



for i in range(4):

    x = df_trn.query('meter == @i')

    

    df_ts['d'].append(x.groupby('ts_day_of_month')['target'].mean())

    df_ts['m'].append(x.groupby('ts_month')['target'].mean())

    df_ts['h'].append(x.groupby('ts_hour')['target'].mean())

    df_ts['w'].append(x.groupby('ts_day_of_week')['target'].mean())



    

for k,d in df_ts.items():

    for i,df in enumerate(d):

        df.plot(label=i)

    plt.title(f'{k}')

    plt.legend()

    plt.show()

    

df_tst = get_data('test')

df_tst = get_ts_features(df_tst)
df_sub = pd.read_csv(f'{DATA}sample_submission.csv', index_col='row_id')

_ = reduce_mem_usage(df_sub)
df_tst.head()
# iterate over meter types:

for j in range(4):

    is_trn_j = (df_trn['meter'] == j)

    is_tst_j = (df_tst['meter'] == j)



    gb_cols = ['building_id', 'ts_month', 'ts_hour', 'ts_day_of_week']

    # Naive model: get mean meter counts on the "training" subset

    df_naive = df_trn[is_trn_j].groupby(gb_cols)['target'].mean()



    # Apply naive model on the validation subset

    x = df_tst[is_tst_j].merge(df_naive.rename('preds').to_frame(), right_index=True, left_on=gb_cols, how='left')



    # fill missing values with the average in the training data

    x['preds'].fillna(df_trn[is_trn_j]['target'].mean(), inplace=True)



    # fill in the submission

    df_sub.loc[x['row_id']] = x['preds'].values.reshape(-1,1)
df_sub = np.expm1(df_sub)
df_sub.to_csv('sub.csv', float_format='%.2f')