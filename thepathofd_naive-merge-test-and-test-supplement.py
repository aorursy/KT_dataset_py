import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
import pickle
import os
import gc
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier,Pool
import lightgbm as lgb
import xgboost as xgb
import matplotlib.pyplot as plt

local_path='../input/'

import pandas as pd

def load_data(name,skip=None,rows=None):
    ''' Load the csv files into a TimeSeries dataframe with minimal data types to reduce the used RAM space. 
    It also saves the files in parquet file to reduce loading time by a factor of ~10.

    Arg:
    
        -name (str): ante_day, last_day, train, train_sample or test

    Returns:
        pd.DataFrame, with int index equal to 'click_id'
    '''

    # Setting file path
    file_path='{}{}'.format(local_path,name)
    if skip!=None:
        skip=range(1,skip)

    # Defining dtypes
    types = {
            'ip':np.uint32,
            'app': np.uint16,
            'os': np.uint16,
            'device': np.uint16,
            'channel':np.uint16,
            'click_time': object
            }

    if name=='test':
        types['click_id']= np.uint32
    elif name=='test_supplement':
        types['click_id']= np.uint32
    else:
        types['is_attributed']='bool'

    # Defining csv file reading parameters
    read_args={
        'nrows':rows,
        'skiprows': skip,
        'parse_dates':['click_time'],
        'infer_datetime_format':True,
        'index_col':'click_time',
        'usecols':list(types.keys()),
        'dtype':types,
        'engine':'c',
        'sep':','
        }

    print('Loading {}.csv'.format(file_path))
    with open('{}.csv'.format(file_path),'rb') as File:
        data=(pd
            .read_csv(File,**read_args)
            .tz_localize('UTC')
            .tz_convert('Asia/Shanghai')
        )

    return data

def force_list(*arg):
    ''' Takes a list of arguments and returns the same, 
    but where all items were forced to a list.

    example : list_1,list_2=force_list(item1,item2)
    '''
    Gen=(x if isinstance(x,list) else [x] for x in arg)
    if len(arg)>1:
        return Gen
    else:
        return next(Gen)
test=load_data('test')
print('The test set has {} observations'.format(len(test)))
test.resample('20T').app.count().plot.bar(figsize=(15,7))
plt.gcf().autofmt_xdate()
test.head()
test_plus=load_data('test_supplement')
print('The test supplement set has {} observations'.format(len(test_plus)))
test_plus.resample('20T').app.count().plot.bar(figsize=(15,7))
plt.gcf().autofmt_xdate()
def combine(test,test_plus):
    test_plus=test_plus.assign(click_id=-1)
    test1=test.loc[test.index.hour.isin([12,13,14]),:]
    test2=test.loc[test.index.hour.isin([17,18,19]),:]
    test3=test.loc[test.index.hour.isin([21,22,23]),:]
    one_sec=pd.Timedelta('1s')
    test_plus1=test_plus.loc[:test1.index.min()-one_sec,:]
    test_plus2=test_plus.loc[test1.index.max()+one_sec:test2.index.min()-one_sec,:]
    test_plus3=test_plus.loc[test2.index.max()+one_sec:test3.index.min()-one_sec,:]
    test_plus4=test_plus.loc[test3.index.max()+one_sec:,:]
    del(test,test_plus)
    gc.collect()
    new_test=pd.concat([test_plus1,test1,test_plus2,test2,test_plus3,test3,test_plus4])
    return new_test.reset_index()
test_whole=combine(test,test_plus)
print('The test supplement set has {} observations'.format(len(test_whole)))
test_plus.resample('20T').app.count().plot.bar(figsize=(15,7))
plt.gcf().autofmt_xdate()
