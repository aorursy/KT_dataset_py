# Uncomment behind to install the package

# !pip install pandas-profiling
import gc

import pandas_profiling



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt



from time import time

from pandas.api.types import is_datetime64_any_dtype as is_datetime

from pandas.api.types import is_categorical_dtype
def reduce_mem_usage(df, use_float16=False, log=False):

    """ Iterate through all the columns of a dataframe and modify the data type to reduce memory usage """



    start_mem = df.memory_usage().sum() / 1024**2

    if log: print("Memory usage of dataframe is {:.2f} MB".format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]) or is_categorical_dtype(df[col]):      # manage categorical columns

            continue

        col_type = df[col].dtype

        

        if col_type != object:

            c_min, c_max = df[col].min(), df[col].max()

            if str(col_type)[:3] == "int":                             # manage columns of type int

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    df[col] = df[col].astype(np.int8)

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)  

            else:                                                     # manage columns of type float

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype("category")



    end_mem = df.memory_usage().sum() / 1024**2

    if log: print("Memory usage after optimization is: {:.2f} MB".format(end_mem))

    if log: print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem)/start_mem))

    

    return df
start = time()



print('Loading csvs \n')

building      = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_test  = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

train         = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

test          = pd.read_csv('../input/ashrae-energy-prediction/test.csv')



print('Reducing memory usage for train and test \n')

train = reduce_mem_usage(train, use_float16=True)

test = reduce_mem_usage(test, use_float16=True)



print('Merging train and test datasets with building dataset \n')

train = train.merge(building, on='building_id', how='left')

test = test.merge(building,   on='building_id', how='left')



print('Merging train and test datasets with weather dataset \n')

train = train.merge(weather_train, on=['site_id', 'timestamp'], how='left')

test = test.merge(weather_test,    on=['site_id', 'timestamp'], how='left')



print('gc \n')

del weather_train, weather_test, building

gc.collect()



print(round(time()-start, 1), ' seconds elapsed')
train_small = train.sample(int(1e5))

test_small = test.sample(int(1e5))
pandas_profiling.ProfileReport(train_small)
pandas_profiling.ProfileReport(test_small)