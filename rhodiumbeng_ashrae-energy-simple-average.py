import numpy as np

import pandas as pd

from itertools import product

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import datetime as dt

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def reduce_mem_usage(df, force_obj_in_category=True, debug=True):

    """ 

    Iterate through all the columns of a dataframe and modify the data type to reduce memory usage. 

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

                df[col] = df[col].astype(np.float32)

        elif force_obj_in_category and 'datetime' not in col_type.name:

            df[col] = df[col].astype('category')



    if debug:

        end_mem = df.memory_usage(deep=True).sum() / 1024**2

        print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

        print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    

    return df
# load training data from csv files

train = pd.read_csv('../input/ashrae-energy-prediction/train.csv')

test = pd.read_csv('../input/ashrae-energy-prediction/test.csv')

building = pd.read_csv('../input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('../input/ashrae-energy-prediction/weather_train.csv')

weather_test = pd.read_csv('../input/ashrae-energy-prediction/weather_test.csv')

submission = pd.read_csv('../input/ashrae-energy-prediction/sample_submission.csv')
print('Shape of the data:','\n','  train_csv: ',train.shape,'\n','  test_csv: ',test.shape,'\n',

      '  building_metadata.csv: ',building.shape,'\n','  weather_train.csv: ',weather_train.shape,'\n',

      '  weather_test.csv: ',weather_test.shape,'\n','  sample_submission.csv: ',submission.shape)
train = reduce_mem_usage(train)

building = reduce_mem_usage(building)

weather_train = reduce_mem_usage(weather_train)

test = reduce_mem_usage(test)

weather_test = reduce_mem_usage(weather_test)

submission = reduce_mem_usage(submission)
train['meter_reading'] = np.log1p(train['meter_reading'])
meter = train.groupby(['meter','building_id']).mean()

meter
test_df = test.drop('timestamp', axis=1)

test_df.head()
# create an empty dataframe with same rows as test_df

result = pd.DataFrame(columns=['row_id', 'building_id', 'meter', 'meter_reading'])

result.head()
for j in meter.index:

    temp = test_df[test_df['meter']==j[0]]

    temp = temp[temp['building_id']==j[1]]

    reading = meter.loc[(j[0], j[1])].values

    temp['meter_reading'] = reading.item(0)

    result = result.append(temp)
len(result)
result.sort_values('row_id')
submission['meter_reading'] = np.expm1(result['meter_reading'])

submission.to_csv('submission.csv', index=False)
submission.sample(100)