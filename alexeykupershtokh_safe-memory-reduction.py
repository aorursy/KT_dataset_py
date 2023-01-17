import numpy as np

import pandas as pd

import os

import gc
x = pd.Series([1111.5, 1111.9, 30001, 30007]) 

print(x)

print('nunique:', x.nunique(), '\n')



y = x.astype(np.float16)

print(y)

print('nunique:', y.nunique())
x = pd.Series(['a', 'b', 'c'])

y = pd.Series(['a', 'c', 'a'])

print(x.astype('category').cat.codes)

print(y.astype('category').cat.codes)
# safe downcast

def sd(col, max_loss_limit=0.001, avg_loss_limit=0.001, na_loss_limit=0, n_uniq_loss_limit=0, fillna=0):

    """

    max_loss_limit - don't allow any float to lose precision more than this value. Any values are ok for GBT algorithms as long as you don't unique values.

                     See https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Precision_limitations_on_decimal_values_in_[0,_1]

    avg_loss_limit - same but calculates avg throughout the series.

    na_loss_limit - not really useful.

    n_uniq_loss_limit - very important parameter. If you have a float field with very high cardinality you can set this value to something like n_records * 0.01 in order to allow some field relaxing.

    """

    is_float = str(col.dtypes)[:5] == 'float'

    na_count = col.isna().sum()

    n_uniq = col.nunique(dropna=False)

    try_types = ['float16', 'float32']



    if na_count <= na_loss_limit:

        try_types = ['int8', 'int16', 'float16', 'int32', 'float32']



    for type in try_types:

        col_tmp = col



        # float to int conversion => try to round to minimize casting error

        if is_float and (str(type)[:3] == 'int'):

            col_tmp = col_tmp.copy().fillna(fillna).round()



        col_tmp = col_tmp.astype(type)

        max_loss = (col_tmp - col).abs().max()

        avg_loss = (col_tmp - col).abs().mean()

        na_loss = np.abs(na_count - col_tmp.isna().sum())

        n_uniq_loss = np.abs(n_uniq - col_tmp.nunique(dropna=False))



        if max_loss <= max_loss_limit and avg_loss <= avg_loss_limit and na_loss <= na_loss_limit and n_uniq_loss <= n_uniq_loss_limit:

            return col_tmp



    # field can't be converted

    return col





def reduce_mem_usage_sd(df, deep=True, verbose=False, obj_to_cat=False):

    numerics = ['int16', 'uint16', 'int32', 'uint32', 'int64', 'uint64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2

    for col in df.columns:

        col_type = df[col].dtypes



        # collect stats

        na_count = df[col].isna().sum()

        n_uniq = df[col].nunique(dropna=False)

        

        # numerics

        if col_type in numerics:

            df[col] = sd(df[col])



        # strings

        if (col_type == 'object') and obj_to_cat:

            df[col] = df[col].astype('category')

        

        if verbose:

            print(f'Column {col}: {col_type} -> {df[col].dtypes}, na_count={na_count}, n_uniq={n_uniq}')

        new_na_count = df[col].isna().sum()

        if (na_count != new_na_count):

            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost na values. Before: {na_count}, after: {new_na_count}')

        new_n_uniq = df[col].nunique(dropna=False)

        if (n_uniq != new_n_uniq):

            print(f'Warning: column {col}, {col_type} -> {df[col].dtypes} lost unique values. Before: {n_uniq}, after: {new_n_uniq}')



    end_mem = df.memory_usage(deep=deep).sum() / 1024 ** 2

    percent = 100 * (start_mem - end_mem) / start_mem

    if verbose:

        print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))

    return df



# https://www.kaggle.com/kyakovlev/ieee-data-minification

def reduce_mem_usage(df, verbose=True):

    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']

    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 # just added 

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

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    percent = 100 * (start_mem - end_mem) / start_mem

    print('Mem. usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(start_mem, end_mem, percent))

    return df
def get_stats(df):

    stats = pd.DataFrame(index=df.columns, columns=['na_count', 'n_unique', 'type', 'memory_usage'])

    for col in df.columns:

        stats.loc[col] = [df[col].isna().sum(), df[col].nunique(dropna=False), df[col].dtypes, df[col].memory_usage(deep=True, index=False) / 1024**2]

    stats.loc['Overall'] = [stats['na_count'].sum(), stats['n_unique'].sum(), None, df.memory_usage(deep=True).sum() / 1024**2]

    return stats



def print_header():

    print('col         conversion        dtype    na    uniq  size')

    print()

    

def print_values(name, conversion, col):

    template = '{:10}  {:16}  {:>7}  {:2}  {:6}  {:1.2f}MB'

    print(template.format(name, conversion, str(col.dtypes), col.isna().sum(), col.nunique(dropna=False), col.memory_usage(deep=True, index=False) / 1024 ** 2))
tmp = pd.read_pickle('/kaggle/input/concat-dataframes/concat.pkl')
tmp.sample(20).sort_index()
tmp.loc['test'].sample(10).sort_index()
# cache a mini-dataset for examples

example = tmp[['card1', 'TransactionAmt', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14']].copy()
stats = get_stats(tmp)

stats
tmp1 = reduce_mem_usage_sd(tmp.copy(), verbose=True)
stats1 = get_stats(tmp1)
tmp1.to_pickle('safe_memreduced_2.5gb.pkl')
tmp1 = reduce_mem_usage_sd(tmp1, verbose=True, obj_to_cat=True)
tmp1.to_pickle('safe_memreduced_1gb.pkl')
# don't copy, as the original df is not needed anymore

tmp2 = reduce_mem_usage(tmp, verbose=True)
overall_stats = pd.concat([stats.add_prefix('a_'), stats1.add_prefix('b_'), get_stats(tmp1).add_prefix('c_')], axis=1)
overall_stats
new_feature = (example.groupby('card1')['TransactionAmt'].transform('mean'))



print_header()

print_values('mean_amt', 'original', new_feature)
new_feature2 = sd(new_feature)



print_header()

print_values('mean_amt', 'default sd():', new_feature2)
new_feature3 = sd(new_feature, n_uniq_loss_limit=100)



print_header()

print_values('mean_amt', 'allow uniq loss:', new_feature3)
new_feature = (example.groupby('card1')['TransactionAmt'].transform('nunique'))

new_feature2 = sd(new_feature)



print_header()

print_values('nunique', 'original', new_feature)

print_values('nunique', 'default sd():', new_feature2)
print_header()

for col in ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14']:

    f = example[col]

    print_values(col, 'original', f)

    

    # here we try to use default setting (precise enough)

    f = sd(f)

    print_values(col, 'default sd():', f)



    # here we allow to fill up to 3 na fields with -99

    f = sd(f, na_loss_limit=3, fillna=-99)

    print_values(col, 'limited na loss:', f)

    print()
