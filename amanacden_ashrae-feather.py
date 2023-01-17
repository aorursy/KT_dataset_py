# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
"""

building = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

sample = pd.read_csv('/kaggle/input/ashrae-energy-prediction/sample_submission.csv')

testW = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

trainW = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

"""
from pandas.api.types import is_datetime64_any_dtype as is_datetime



def reduce_mem_usage(df, use_float16=False):

    """ iterate through all the columns of a dataframe and modify the data type

        to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    

    for col in df.columns:

        if is_datetime(df[col]):

            # skip datetime type

            continue

        col_type = df[col].dtype

        

        if col_type != object:

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

                if use_float16 and c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    df[col] = df[col].astype(np.float16)

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')



    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df





def import_data(file):

    """create a dataframe and optimize its memory usage"""

    df = pd.read_csv(file, parse_dates=True, keep_date_col=True)

    df = reduce_mem_usage(df)

    return df
%%time



# Read data...

root = '../input/ashrae-energy-prediction'



train_df = pd.read_csv(os.path.join(root, 'train.csv'))

weather_train_df = pd.read_csv(os.path.join(root, 'weather_train.csv'))

test_df = pd.read_csv(os.path.join(root, 'test.csv'))

weather_test_df = pd.read_csv(os.path.join(root, 'weather_test.csv'))

building_meta_df = pd.read_csv(os.path.join(root, 'building_metadata.csv'))

sample_submission = pd.read_csv(os.path.join(root, 'sample_submission.csv'))
train_df['timestamp'] = pd.to_datetime(train_df['timestamp'])

test_df['timestamp'] = pd.to_datetime(test_df['timestamp'])

weather_train_df['timestamp'] = pd.to_datetime(weather_train_df['timestamp'])

weather_test_df['timestamp'] = pd.to_datetime(weather_test_df['timestamp'])
reduce_mem_usage(train_df)

reduce_mem_usage(test_df)

reduce_mem_usage(building_meta_df)

reduce_mem_usage(weather_train_df)

reduce_mem_usage(weather_test_df)
%%time



train_df.to_feather('train.feather')

test_df.to_feather('test.feather')

weather_train_df.to_feather('weather_train.feather')

weather_test_df.to_feather('weather_test.feather')

building_meta_df.to_feather('building_metadata.feather')

sample_submission.to_feather('sample_submission.feather')