# Example of parallel memory optimization of pandas Dataframes. Inspired by a.o. https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
# Import packages

import pandas as pd

import numpy as np

from joblib import Parallel, delayed # For multiprocessing

import multiprocessing  # For multiprocessing



# Import data

df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv', index_col='TransactionID')
def reduce_mem(df):

    # Old memory footprint

    mem_old = df.memory_usage().sum() / (1024 ** 2)

    # Retrieve number of cores for multiprocessing

    num_cores = multiprocessing.cpu_count()

    # Resulting dataframes

    res = Parallel(n_jobs = num_cores, max_nbytes=None)(delayed(reduce_mem_single)(df.loc[:, column]) for column in df.columns)

    # Concatenate results       

    df_new = pd.concat(res, axis=1)    

    # New footprint

    mem_new = df_new.memory_usage().sum() / (1024 ** 2)

    # Reduction

    red = mem_new / mem_old - 1

    print('Dataframe memory reduction: ' +str(np.round(red * 100, 2)) + '%')

    

    return df_new

    

def reduce_mem_single(col):

       

    if col.dtype == 'object':               # random choice of setting to category...

        if col.nunique() < 20000:

            col = col.astype('category')           

    elif col.dtype.name[0] in ('f', 'i'):      # super hacky way of quickly checking if dtype is float or int 

        # Set NaN to -999, better would be a value outside of the current range (i.e. below min or above max), but this is more convenient for later on

        col = col.fillna(int(-999)) 



        # max and min value

        minimum = int(col.min())

        maximum = int(col.max())



        # Determine if there are values with decimal places

        dec = ((col % 1) == 0).astype('uint8').sum() != len(col)

        

        # Set as type 

        if dec:

            col = col.astype('float32')

        else:

            if minimum >= 0:

                if maximum <= 255:

                    col = col.astype('uint8')

                elif maximum <= 65535:

                    col = col.astype('uint16')

                elif maximum <= 4294967295:

                    col = col.astype('uint32')

                else:

                    col = col.astype('uint64')

            else:

                if (minimum >= -128) & (maximum <= 127):

                    col = col.astype('int8')

                elif (minimum >= -32768) & (maximum <= 32767):

                    col = col.astype('int16')

                elif (minimum >= -2147483648) & (maximum <= 2147483647):

                    col = col.astype('int32')

                else:

                    col = col.astype('int64')                

     

    return col
# Execute on total dataset and check memory reduction

df = reduce_mem(df)