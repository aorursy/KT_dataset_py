import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
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



def plot_point(df1, df2, col):

    for a in col:

        colors1 = '#00CED1'

        colors2 = '#DC143C'

        plt.figure(figsize=(12,6))

        plt.grid()

        plt.scatter(df1[df1['isFraud'] == 0]['TransactionDT'], df1[df1['isFraud'] == 0][a], c=colors1, alpha=0.4, s=0.5, label='NoFraud')

        plt.scatter(df1[df1['isFraud'] == 1]['TransactionDT'], df1[df1['isFraud'] == 1][a], c=colors2, alpha=0.8, s=0.5, label='Fraud')

        plt.grid()

        plt.scatter(df2['TransactionDT'], df2[a], c=colors1, alpha=0.4, s=0.5)

        plt.legend()

        plt.show()
train_id = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')

train_trans = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

test_id = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')

test_trans = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
train_id = reduce_mem_usage(train_id)

train_trans = reduce_mem_usage(train_trans)

test_id = reduce_mem_usage(test_id)

test_trans = reduce_mem_usage(test_trans)
index0 = ['D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12','D13','D14','D15']

plot_point(train_trans, test_trans, index0)