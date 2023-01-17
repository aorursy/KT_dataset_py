# General imports

import numpy as np

import pandas as pd

import os, warnings, random, time, pickle



from sklearn.preprocessing import LabelEncoder



warnings.filterwarnings('ignore')
## Memory Reducer

# :df pandas dataframe to reduce size             # type: pd.DataFrame()

# :verbose                                        # type: bool

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
########################### DATA LOAD

#################################################################################

print('Load Data')

train_df = pd.read_csv('../input/ieee-fraud-detection/train_transaction.csv')

train_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv')
########################### Concat vs Merge

#################################################################################



# Let's try to merge data on TransactionID

start = time.time()

df = train_df.merge(train_identity, on=['TransactionID'], how='left')

del df

print('Merging time:', time.time()-start)



# Can we do it faster?



start = time.time()

df = train_df[['TransactionID']]

df = df.merge(train_identity, on=['TransactionID'], how='left')

del df['TransactionID']

df = pd.concat([train_df, df], axis=1)

del df

print('Concat time:', time.time()-start)



# Why is it so?

# Merging is more memory consuming

# Merging is just slowly than concat

# Because train_df is huge and has 394 columns 

# and we are putting all in "temp var" on merging
########################### Map vs Merge

#################################################################################



# Lets try to find mean TransactionID by card1 and the merge it to train_df

start = time.time()



df = train_df.groupby(['card1'])['TransactionID'].agg(['mean']).reset_index()

df = train_df.merge(df, on=['card1'], how='left')

del df



print('Aggregation + Merging time:', time.time()-start)



# Lets try to use map

start = time.time()



temp_dict = train_df.groupby(['card1'])['TransactionID'].agg(['mean']).reset_index()

temp_dict.index = temp_dict['card1'].values

temp_dict = temp_dict['mean'].to_dict()



df = train_df.copy()

df['mean'] = df['card1'].map(temp_dict)

del temp_dict, df



print('Aggregation + Mapping time:', time.time()-start)



# Why is it so?

# Merging is more memory consuming

# Merging is just slowly than mapping on single column
########################### Int vs Float

#################################################################################



df = train_df[['card1','card2','card3','card5',]]

df.info()

# Base memory usage: 18.0 MB



# df = df.astype(int) will give error 

# as this pandas version can't convert nans to int

df2 = df.fillna(-1).astype(int) 

df2.info()

# Base memory usage: 18.0 MB



# Same 18.0 MB. Obviously it is because of byte that dtype has 64)))

# https://docs.scipy.org/doc/numpy-1.13.0/user/basics.types.html



# Let's reduce type

df2 = df.astype(np.float32)

df2.info()

# Base memory usage: 9.0 MB (f(64) -> 18MB / f(32) -> 9MB -> obviously)



# Problem with floats here is with precision(mantissa)

our_float = 0.12345678901234567890



# float64 -> Double precision float: sign bit, 11 bits exponent, 52 bits mantissa

print('New float:', np.float64(our_float))

print('Is it same?:', our_float==np.float64(our_float))



# float32 -> Single precision float: sign bit, 8 bits exponent, 23 bits mantissa

print('New float:', np.float32(our_float))

print('Is it same?:', our_float==np.float32(our_float))



# Ups we have different number -> be carefull with it

# Ints we can convert without this fear (based only on type max/min values)
########################### String vs Category

#################################################################################



df = train_df[['card1','card2','card3','card5',]]

df.info()

# Base memory usage: 18.0 MB



df2 = df.astype('str')

df2.info()

# Base memory usage more than floats



df2 = df.astype('category')

df2.info()

# Base memory usage: 4.1 MB



# Why so?

# Beacause Pandas category is a type for "Finite list of text values"

# We will not reserve memory for it