# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
# util functions to reduce pandas dataframe memory
def df_mem_reduce(df, verbose=True):
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

import pickle

#util function to dump any object as pickled file
# ANOTHER APPROACH: pickle.dump(file_, open(filename+'.pkl','wb'), pickle.HIGHEST_PROTOCOL)
def dump(file_, filename):
    with open(filename+'.pkl','wb') as f:
        pickle.dump(file_, f, pickle.HIGHEST_PROTOCOL)

#util function to load any pickled file dumped by above function
# ANOTHER APPROACH: pickle.load(open(filename,'rb'))
def load(filename):
    with open(filename,'rb') as f:
        return pickle.load(f)



sell = pd.read_csv('../input/m5-forecasting-accuracy/sell_prices.csv')
print("%6.2f Mb" % (sell.memory_usage().sum() / 1024**2))
print(sell.dtypes)
sell_reduced = reduce_mem_usage(sell)
print(sell_reduced.dtypes)
print("%6.2f Mb" % (sell_reduced.memory_usage().sum() / 1024**2))
sell_reduced.memory_usage()
import csv
import gzip
import json

#util function to read csv file as list of dictionary
def read_csv_as_list(filename_to_read):
    """
    filename_to_read: filename with path to read
    """
    return [x for x in csv.DictReader(open(filename_to_read,'r'))] # file_ = '*.csv'
    

#util function to write data in list of dictionary format as csv file
def write_list_as_csv(inputList, filename_to_write, columns):
    """
    inputList: input list of dictionaries to dump
    filename_to_write: filename with path to write
    columns: comma separated string containing column names to write in header
    """
    headerDict = OrderedDict([(x, None) for x in columns.split(',')])
    with open(filename_to_write,'wb') as fout:
        dw = csv.DictWriter(fout, delimiter=',',fieldnames=headerDict, quoting =csv.QUOTE_ALL)
        dw.writeheader()
        for item in inputList:
            dw.writerow(item)


def write_list_as_zipped_csv(inputList, filename_to_write):
    """
    inputList: input list of dictionaries to dump
    filename_to_write: filename with path to write
    columns: comma separated string containing column names to write in header
    """
    with gzip.open(filename_to_write,'w') as outFile:
        for lJ in inputList:
            row = json.dumps(lJ)
            print(row)
            outFile.write(row.encode())
            outFile.write('\n')
            

sell_list = read_csv_as_list(filename_to_read='../input/m5-forecasting-accuracy/sell_prices.csv')
print("%6.2f Mb" % (sell_list.__sizeof__() / 1024**2))
sell_list[0]