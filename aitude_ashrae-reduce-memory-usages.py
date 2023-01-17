import os

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
DATA_PATH = '../input/ashrae-energy-prediction/'

building_df = pd.read_csv(DATA_PATH + 'building_metadata.csv')

train_df = pd.read_csv(DATA_PATH + 'train.csv')

test_df = pd.read_csv(DATA_PATH + 'test.csv')

weather_train_df = pd.read_csv(DATA_PATH + 'weather_train.csv')

weather_test_df = pd.read_csv(DATA_PATH + 'weather_test.csv')
def ds_size(dataset):

    return (dataset.memory_usage().sum()) / (1024 ** 2)



def ds_statistics(df_names,df_lists) :

    datasets_df = pd.DataFrame(columns=['Dataset','Number of Rows','Number of Columns','Size (MB)'])

    rows = [df.shape[0] for df in df_lists]

    columns = [df.shape[1] for df in df_lists]

    size_MB = [ds_size(df) for df in df_lists]



    datasets_df['Dataset'] = df_names

    datasets_df['Number of Rows'] = rows

    datasets_df['Number of Columns'] = columns

    datasets_df['Size (MB)'] = size_MB

    datasets_df = datasets_df.set_index('Dataset')

    datasets_df = datasets_df.sort_values('Size (MB)')

    return datasets_df



def ds_optimization(dataframe,categorical = []):

    df = dataframe.copy()

    int_types = {np.int8 : (np.iinfo(np.int8).min,np.iinfo(np.int8).max),

                 np.int16: (np.iinfo(np.int16).min,np.iinfo(np.int16).max),

                 np.int32 : (np.iinfo(np.int32).min,np.iinfo(np.int32).max),

                 np.int64 : (np.iinfo(np.int64).min,np.iinfo(np.int64).max)

                }

    float_types = {np.float16: (np.finfo(np.float16).min,np.finfo(np.float16).max), 

                   np.float32: (np.finfo(np.float32).min,np.finfo(np.float32).max), 

                   np.float64: (np.finfo(np.float64).min,np.finfo(np.float64).max)

                  }

    for col in df.columns:

        col_type = df[col].dtypes

        col_min = df[col].min()

        col_max = df[col].max()

        if (str(col_type)[:3] == 'int') & (str(col) in categorical):

            df[col] = pd.Categorical(df[col])

        elif str(col_type)[:3] == 'int':

            for dtype,drange in int_types.items():

                if (col_min > drange[0]) & (col_max < drange[1]):

                    df[col] = df[col].astype(dtype)

                    break

        elif str(col_type)[:5] == 'float':

            for dtype,drange in float_types.items():

                if (col_min > drange[0]) & (col_max < drange[1]):

                    df[col] = df[col].astype(dtype)

                    break

    return df
dataset_name = ['building','train','test','weather_train','weather_test']



origianl_ds = ds_statistics(dataset_name,[building_df,train_df,test_df,weather_train_df,weather_test_df])

print(origianl_ds.head())
optimized_ds = ds_statistics(dataset_name,[ds_optimization(building_df),ds_optimization(train_df,['meter']),ds_optimization(test_df),ds_optimization(weather_train_df),ds_optimization(weather_test_df)])

print(optimized_ds.head())
optimized_percentage = 100 * (origianl_ds['Size (MB)'] - optimized_ds['Size (MB)'])/origianl_ds['Size (MB)']

print(optimized_percentage)
plt.figure(figsize=(10, 5))  # width:20, height:3

plt.title('Datasets Memory Decreased (%)')

plt.xlabel('Dataset')

plt.ylim([0,100])

plt.ylabel('% Changes')

plt.bar(optimized_percentage.index,optimized_percentage.values,align='center')

plt.show()