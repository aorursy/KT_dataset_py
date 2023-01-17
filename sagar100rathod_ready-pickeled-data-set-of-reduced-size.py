

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import pickle



print(os.listdir("../input/ieee-fraud-detection"))



# Any results you write to the current directory are saved as output.
datadir = "../input/ieee-fraud-detection"

train_identity_df = pd.read_csv(os.path.join(datadir, 'train_identity.csv'))

train_transaction_df = pd.read_csv(os.path.join(datadir, 'train_transaction.csv'))



print('Train Idenitity Shape:', train_identity_df.shape)

print('Train Transaction Shape:', train_transaction_df.shape)

test_identity_df = pd.read_csv(os.path.join(datadir, 'test_identity.csv'))

test_transaction_df = pd.read_csv(os.path.join(datadir, 'test_transaction.csv'))



print('Test Identity Shape:', test_identity_df.shape)

print('Test Transaction Shape:', test_transaction_df.shape)
def mem_usage(pandas_obj):

    if isinstance(pandas_obj,pd.DataFrame):

        usage_b = pandas_obj.memory_usage(deep=True).sum()

    else: # we assume if not a df it's a series

        usage_b = pandas_obj.memory_usage(deep=True)

    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes

    return "{:03.2f} MB".format(usage_mb)



def convert_type(df, column, source_type, target_type):

    print('Memory Usage for ({},{}) Before:{}'.format(column, source_type, mem_usage(df[column])), end=' ')

    df[column] = df[column].astype(target_type)

    print('------->({},{}) After:{}'.format(column, target_type, mem_usage(df[column])))
def reduce_memory(df):

    print("Reduce_memory...");

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            target_type = None

            if str(col_type)[:3] == 'int':

                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:

                    target_type = np.int8

                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    target_type = np.int16

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    target_type = np.int32

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    target_type = np.int64

            else:

                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:

                    target_type = np.float16

                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    target_type = np.float32

                else:

                    target_type = np.float64

            convert_type(df, col, col_type, target_type)

        else:

            # for object types

            # check number of unique values

            unique_values = df[col].nunique()

            if unique_values / len(df[col]) < 0.5:

                convert_type(df, col, col_type, 'category')

    return df
def save_df_as_pickle(df, filename):

    with open(filename, 'wb') as outfile:

        pickle.dump(df, outfile)

    statinfo = os.stat(filename)

    print('{} file size:'.format(filename),statinfo.st_size / (1024 ** 2), 'MB')
print('Memory Usage Before:', mem_usage(train_identity_df))

train_identity_df = reduce_memory(train_identity_df)

print('Memory Usage After:', mem_usage(train_identity_df))

save_df_as_pickle(train_identity_df, 'train_identity_df.pkl')
print('Memory Usage Before:', mem_usage(train_transaction_df))

train_transaction_df = reduce_memory(train_transaction_df)

print('Memory Usage After:', mem_usage(train_transaction_df))

save_df_as_pickle(train_transaction_df, 'train_transaction_df.pkl')
print('Memory Usage Before:', mem_usage(test_identity_df))

test_identity_df = reduce_memory(test_identity_df)

print('Memory Usage After:', mem_usage(test_identity_df))

save_df_as_pickle(test_identity_df, 'test_identity_df.pkl')
print('Memory Usage Before:', mem_usage(test_transaction_df))

test_transaction_df = reduce_memory(test_transaction_df)

print('Memory Usage After:', mem_usage(test_transaction_df))

save_df_as_pickle(test_transaction_df, 'test_transaction_df.pkl')
## verify files are present

os.listdir()