import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import pyarrow as pa

import dask
import dask.dataframe as dd

# Set up a logger to dump messages to both log file and notebook
import logging as logging
def ini_log(filename):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    
    handlers = [logging.StreamHandler(None), logging.FileHandler(filename, 'a')]
    
    fmt=logging.Formatter('%(asctime)-15s: %(levelname)s  %(message)s')
    for h in handlers:
        h.setFormatter(fmt)
        logger.addHandler(h)
    return logger
        
log = ini_log('out.log')
#log.basicConfig(filename='out.log',level=log.DEBUG, format='%(asctime)-15s: %(levelname)s  %(message)s')

import gc
gc.enable()
def_num = np.nan
def_str = 'NaN'

def get_keys_for_field(field=None):
    the_dict = {
        'device': [
            'browser', 'object',
            'deviceCategory',
            ('isMobile', False, bool),
            'operatingSystem'
        ],
        'geoNetwork': [
            'city',
            'continent',
            'country',
            'metro',
            'networkDomain',
            'region',
            'subContinent'
        ],
        'totals': [
            ('pageviews', 0, np.int16),
            ('hits', def_num, np.int16),
            ('bounces', 0, np.int8),
            ('newVisits', 0, np.int16),
            ('transactionRevenue', 0, np.int64),
            ('visits', -1, np.int16),
            ('timeOnSite', -1, np.int32),
            ('sessionQualityDim', -1, np.int8),
        ],
        'trafficSource': [
            'adContent',
            #'adwordsClickInfo',
            'campaign',
            ('isTrueDirect', False, bool),
            #'keyword', #can not be saved in train (utf-8 symbols left)
            'medium',
            'referralPath',
            'source'
        ],
    }
    return the_dict[field]


def convert_to_dict(x):
    #print(x, type(x))
    return eval(x.replace('false', 'False')
                .replace('true', 'True')
                .replace('null', 'np.nan'))

def develop_json_fields(fin, json_fields=['totals'], bsize=1e8, cols_2drop=[]):
    df = dd.read_csv(fin, blocksize=bsize, 
                 #converters={column: json.loads for column in JSON_COLUMNS},
                 dtype={'fullVisitorId': 'str', # Important!!
                        #usecols=lambda c: c not in cols_2drop,
                            'date': 'str',
                            **{c: 'str' for c in json_fields}
                           },
                     parse_dates=['date'],)#.head(10000, 100)
    
    df = df.drop(cols_2drop, axis=1)
    
    # Get the keys
    for json_field in json_fields:
        log.info('Doing Field {}'.format(json_field))
        # Get json field keys to create columns
        the_keys = get_keys_for_field(json_field)
        # Replace the string by a dict
        log.info('Transform string to dict')        
        df[json_field] = df[json_field].apply(lambda x: convert_to_dict(x), meta=('','object'))
        
        log.info('{} converted to dict'.format(json_field))
        #display(df.head())
        for k in the_keys:
            if isinstance(k, str):
                t_ = def_str
                k_ = k
            else:
                t_ = k[1]
                k_ = k[0]
            df[json_field + '_' + k_] = df[json_field].to_bag().pluck(k_, default=t_).to_dataframe().iloc[:,0]
            if not isinstance(k, str) and len(k)>2:
                df[json_field + '_' + k_] = df[json_field + '_' + k_].astype(k[2])
            
        del df[json_field]
        gc.collect()
        log.info('{} fields extracted'.format(json_field))
    return df

print(os.listdir("../input"))
!head ../input/train_v2.csv
JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
DROP_COLUMNS = ['customDimensions', 'hits', 'socialEngagementType']

def measure_memory(df, name):
    size_df = df.memory_usage(deep=True)
    log.info('{} size: {:.2f} MB'.format(name, size_df.sum().compute()/ 1024**2))
    
def read_parse_store(fin, label='XXX', bsize=1e9):
    log.debug('Start with {}'.format(label))
    df_  = develop_json_fields(fin,  bsize=bsize, json_fields=JSON_COLUMNS, cols_2drop=DROP_COLUMNS)
    
    #some stats
    measure_memory(df_, label)
    log.info('Number of partitions in {}: {}'.format(label, df_.npartitions))
    
    #visualize a few rows
    display(df_.head())
    
    #reduce var size
    df_['visitNumber'] = df_['visitNumber'].astype(np.uint16)

    #read the whole dataset into pd.DataFrame in memory and store into a single file
    #otherwise dask.DataFrame would be stored into multiple files- 1 per partition
    df_.compute().to_csv("{}-flat.csv.gz".format(label), index=False , compression='gzip')
%%time
read_parse_store('../input/train_v2.csv', 'train')
%%time
read_parse_store('../input/test_v2.csv', 'test')
!ls -l
