import gc

import pickle



import pandas as pd

import numpy as np
train_tr = pd.read_csv("../input/ieee-fraud-detection/train_transaction.csv")

test_tr = pd.read_csv("../input/ieee-fraud-detection/test_transaction.csv")

train_id = pd.read_csv("../input/ieee-fraud-detection/train_identity.csv")

test_id = pd.read_csv("../input/ieee-fraud-detection/test_identity.csv")
train = pd.merge(train_tr, train_id, how='left', on='TransactionID')

test = pd.merge(test_tr, test_id, how='left', on='TransactionID')

print(train.shape, test.shape)
del train_tr, test_tr, train_id, test_id

gc.collect()
def reduce_memory_usage(df):

    """

    iterate through all the columns of a dataframe and modify the data type

    to reduce memory usage.        

    """

    start_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:

        col_type = df[col].dtype

        if col_type != object:

            c_min = df[col].min()

            c_max = df[col].max()

            if str(col_type)[:3] == 'int':   # Integer column

                if c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:

                    df[col] = df[col].astype(np.int16)

                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:

                    df[col] = df[col].astype(np.int32)

                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:

                    df[col] = df[col].astype(np.int64)                

            else:   # Float column

                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:

                    df[col] = df[col].astype(np.float32)

                else:

                    df[col] = df[col].astype(np.float64)

        else:

            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2

    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))

    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
train = reduce_memory_usage(train)

test = reduce_memory_usage(test)

gc.collect()
train.to_csv("Train.csv", index=False)

test.to_csv("Test.csv", index=False)
train_dtype_dict = {}

for i, column in enumerate(train.columns):

    train_dtype_dict[column] = train[column].dtype

test_dtype_dict = {}

for i, column in enumerate(test.columns):

    test_dtype_dict[column] = test[column].dtype
with open("Train-dtypes.pkl", "wb") as f:

    pickle.dump(train_dtype_dict, f)

with open("Test-dtypes.pkl", "wb") as f:

    pickle.dump(test_dtype_dict, f)