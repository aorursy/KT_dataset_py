from  datetime import datetime, timedelta

import gc

import numpy as np, pandas as pd

import lightgbm as lgb



import os, sys, gc, time, warnings, pickle, psutil, random



warnings.filterwarnings('ignore')
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
FIRST_DAY = 710 # If you want to load all the data set it to '1' -->  Great  memory overflow  risk !



grid2_colnm = ['sell_price', 'price_max', 'price_min', 'price_std',

               'price_mean', 'price_norm', 'price_nunique', 'item_nunique',

               'price_momentum', 'price_momentum_m', 'price_momentum_y']



grid3_colnm = ['event_name_1', 'event_type_1', 'event_name_2',

               'event_type_2', 'snap_CA', 'snap_TX', 'snap_WI', 'tm_d', 'tm_w', 'tm_m',

               'tm_y', 'tm_wm', 'tm_dw', 'tm_w_end']



mean_encoding_combination = [

    ['store_id','dept_id'], 

    ['store_id','item_id'], 

    ['store_id', 'dept_id', 'tm_m'],

    ['store_id', 'dept_id', 'tm_m', 'tm_dw'],

    ['store_id', 'dept_id', 'tm_dw' ,'snap_CA'],

    ['store_id', 'dept_id', 'tm_dw' ,'snap_TX'],

    ['store_id', 'dept_id', 'tm_dw' ,'snap_WI']

]
grid_1 = pd.read_pickle("../input/m5-simple-fe/grid_part_1.pkl")

grid_2 = pd.read_pickle("../input/m5-simple-fe/grid_part_2.pkl")[grid2_colnm]

grid_3 = pd.read_pickle("../input/m5-simple-fe/grid_part_3.pkl")[grid3_colnm]



grid_df = pd.concat([grid_1, grid_2, grid_3], axis=1)

del grid_1, grid_2, grid_3; gc.collect()



grid_df = grid_df[(grid_df['d'] >= FIRST_DAY)]



grid_df = reduce_mem_usage(grid_df)
used_cols = list()

for b in mean_encoding_combination:

    for a in b:

        used_cols.append(a)
grid_df = grid_df[['id'] + list(set(used_cols)) + ['sales']]
for col in mean_encoding_combination:

    print(col, 'encoding')

    colnm1 = '_'.join(col)+'_mean_enc'

    colnm2 = '_'.join(col)+'_std_enc'

    grid_df[colnm1] = grid_df.groupby(col)['sales'].transform('mean')

    grid_df[colnm2] = grid_df.groupby(col)['sales'].transform('std')
enc_data = grid_df[['id']+[a for a in grid_df if 'enc' in a]]

enc_data = reduce_mem_usage(enc_data)

enc_data.to_pickle('mean_enc_reg.pkl')
del enc_data, grid_df; gc.collect()
grid_1 = pd.read_pickle("../input/m5-simple-fe/grid_part_1.pkl")

grid_2 = pd.read_pickle("../input/m5-simple-fe/grid_part_2.pkl")[grid2_colnm]

grid_3 = pd.read_pickle("../input/m5-simple-fe/grid_part_3.pkl")[grid3_colnm]



grid_df = pd.concat([grid_1, grid_2, grid_3], axis=1)

del grid_1, grid_2, grid_3; gc.collect()



# for classification transform TARGET to binary

grid_df['sales'] = np.where(grid_df['sales'] == 0 , 0, 1)



sales = grid_df.sales

sales[grid_df.d > 1913] = np.nan

grid_df['sales'] = sales



grid_df = grid_df[(grid_df['d'] >= FIRST_DAY)]



grid_df = reduce_mem_usage(grid_df)
used_cols = list()

for b in mean_encoding_combination:

    for a in b:

        used_cols.append(a)
grid_df = grid_df[['id'] + list(set(used_cols)) + ['sales']]
for col in mean_encoding_combination:

    print(col, 'encoding')

    colnm1 = '_'.join(col)+'_mean_enc'

    colnm2 = '_'.join(col)+'_std_enc'

    grid_df[colnm1] = grid_df.groupby(col)['sales'].transform('mean')

    grid_df[colnm2] = grid_df.groupby(col)['sales'].transform('std')
enc_data = grid_df[['id']+[a for a in grid_df if 'enc' in a]]

enc_data = reduce_mem_usage(enc_data)

enc_data.to_pickle('mean_enc_clf.pkl')