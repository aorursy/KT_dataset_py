import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import gc

gc.enable()

import time

import warnings

warnings.filterwarnings("ignore")
#print(os.listdir("../input"))

!ls -GFlash  ../input/ashrae-energy-prediction
%%time

# import Dataset to play with it

train_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')

building = pd.read_csv('/kaggle/input/ashrae-energy-prediction/building_metadata.csv')

weather_train = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_train.csv')

train_data = train_data.merge(building, on='building_id', how='left')

train_data = train_data.merge(weather_train, on=['site_id', 'timestamp'], how='left')



test_data = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

weather_test = pd.read_csv('/kaggle/input/ashrae-energy-prediction/weather_test.csv')

test_data = test_data.merge(building, on='building_id', how='left')

test_data = test_data.merge(weather_test, on=['site_id', 'timestamp'], how='left')



print ("Done!")
print('Shape of Data:')

print(train_data.shape)

print(test_data.shape)
del building, weather_train, weather_test
train_data.info()
test_data.info()
#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65

def reduce_mem_usage(df):

    start_mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage of properties dataframe is :",start_mem_usg," MB")

    NAlist = [] # Keeps track of columns that have missing values filled in. 

    for col in df.columns:

        if df[col].dtype != object:  # Exclude strings            

            # Print current column type

            print("******************************")

            print("Column: ",col)

            print("dtype before: ",df[col].dtype)            

            # make variables for Int, max and min

            IsInt = False

            mx = df[col].max()

            mn = df[col].min()

            print("min for this col: ",mn)

            print("max for this col: ",mx)

            # Integer does not support NA, therefore, NA needs to be filled

            if not np.isfinite(df[col]).all(): 

                NAlist.append(col)

                df[col].fillna(mn-1,inplace=True)  

                   

            # test if column can be converted to an integer

            asint = df[col].fillna(0).astype(np.int64)

            result = (df[col] - asint)

            result = result.sum()

            if result > -0.01 and result < 0.01:

                IsInt = True            

            # Make Integer/unsigned Integer datatypes

            if IsInt:

                if mn >= 0:

                    if mx < 255:

                        df[col] = df[col].astype(np.uint8)

                    elif mx < 65535:

                        df[col] = df[col].astype(np.uint16)

                    elif mx < 4294967295:

                        df[col] = df[col].astype(np.uint32)

                    else:

                        df[col] = df[col].astype(np.uint64)

                else:

                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:

                        df[col] = df[col].astype(np.int8)

                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:

                        df[col] = df[col].astype(np.int16)

                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:

                        df[col] = df[col].astype(np.int32)

                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:

                        df[col] = df[col].astype(np.int64)    

            # Make float datatypes 32 bit

            else:

                df[col] = df[col].astype(np.float32)

            

            # Print new column type

            print("dtype after: ",df[col].dtype)

            print("******************************")

    # Print final result

    print("___MEMORY USAGE AFTER COMPLETION:___")

    mem_usg = df.memory_usage().sum() / 1024**2 

    print("Memory usage is: ",mem_usg," MB")

    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")

    return df, NAlist
train_data, NAlist = reduce_mem_usage(train_data)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
test_data, NAlist = reduce_mem_usage(test_data)

print("_________________")

print("")

print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")

print("_________________")

print("")

print(NAlist)
train_data.info()
test_data.info()
#train_data.to_csv('train.csv', index=False)

#test_data.to_csv('test.csv', index=False)
train_data.to_pickle("train_data.pkl")

print("train_data size after memory reduction:", os.stat('train_data.pkl').st_size * 1e-6)





test_data.to_pickle("test_data.pkl")

print("test_data size after memory reduction:", os.stat('test_data.pkl').st_size * 1e-6)