import numpy as np

import pandas as pd



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# =================================

pd.set_option('display.max_columns', 500)



import pandas_profiling

from sklearn.preprocessing import LabelEncoder

import xgboost as xgb



## thanks to Thien for this code which was taken from https://www.kaggle.com/suoires1/fraud-detection-eda-and-modeling



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
%%time

data_dir = "/kaggle/input/ieee-fraud-detection/"

files = ["train_identity.csv", "train_transaction.csv", "test_identity.csv", "test_transaction.csv"]

df_train_ident, df_train_trans, df_test_ident, df_test_trans = [

    pd.read_csv(os.path.join(data_dir, f), index_col='TransactionID') for f in files]
%%time

# join transactions with identities and validate no transaction is lost

df_train = df_train_trans.join(df_train_ident, how="left")

assert df_train_trans.shape[0] == df_train.shape[0]



df_test = df_test_trans.join(df_test_ident, how="left")

assert df_test_trans.shape[0] == df_test.shape[0]
%%time

# optimising memory

df_train=reduce_mem_usage(df_train)

df_test=reduce_mem_usage(df_test)

# export it

df_train.to_pickle("train.pkl")

df_test.to_pickle("test.pkl")
for col in df_train.select_dtypes('object').columns:

    no_new_cats = len(set(df_test[col].unique())-set(df_train[col].unique()))

    if no_new_cats > 0:

        print(f"Column {col} has {no_new_cats} new categories in test set")
