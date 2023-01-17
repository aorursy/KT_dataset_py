# data engineering libraries
import numpy as np 
import pandas as pd 

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sns

# support python libraries
import warnings
import time
import sys
import datetime

# machine learning libraries
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.set_option('display.max_columns', 500)
def reduce_mem_usage(df, verbose=True):
    """
    Reduce dataset memory usage by changing the dtypes within each dataset column
    :param df: (pd.DataFrame) dataset to be changed
    :param verbose: (bool) Flag indicating if we should verbose actions
    
    :return: (pd.DataFrame) optimized dataset
    """
    # list numeric datatypes
    numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    
    # set up the initial memory usage
    start_mem = df.memory_usage().sum() / 1024**2   
    
    # for each column in the dataset
    for col in df.columns:
        # get the column type
        col_type = df[col].dtypes
        
        # if the column type is within the numeric datatypes
        if col_type in numerics:
            # calculate the column value range
            c_min = df[col].min()
            c_max = df[col].max()
            
            # Change the column type based on its range
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.uint8).min and c_max <= np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64) 
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64) 
        # if the column is not of numeric type
        elif col_type == np.object:
            # change it to categorical
            df[col] = df[col].astype('category')
                
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: 
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df
def dataset_overview(df):
    """
    """
    dataset_info = list()
    for c in df.columns:
        # create a list of column information to add
        column_info = list()
        
        # add the column name
        column_info.append(c)
        
        # add the column type
        column_info.append(df[c].dtypes)
        
        # add the column memory usage
        column_info.append('%.2f' % (df[c].memory_usage(index=False, deep=True)/(1024*1024)))
        
        # add the percentage of fill
        column_info.append('%.1f%%' % (100*df[c].count()/df.shape[0]))
        
        # add the number of unique values
        column_info.append(df[c].nunique())
        
        # add the example of values
        column_info.append(df[c].unique())
        
        # add the column info to the dataset info
        dataset_info.append(column_info)
        
    # return a dataframe with all information
    return pd.DataFrame(data=dataset_info, columns=['Column Name', 'Column Type', 'Memory Usage (MB)', '% of Fill', 'Unique Values', 'Examples'])
        
rNewTransc = pd.read_csv('../input/new_merchant_transactions.csv', parse_dates=['purchase_date'])
dataset_overview(rNewTransc)
rHistTransc = pd.read_csv('../input/historical_transactions.csv', parse_dates=['purchase_date'])
dataset_overview(rHistTransc)
rMerchant = pd.read_csv('../input/merchants.csv', parse_dates=['first_active_month'])
dataset_overview(rMerchant)
rTrain = pd.read_csv('../input/train.csv', parse_dates=['first_active_month'])
dataset_overview(rTrain)
rTest = pd.read_csv('../input/test.csv', parse_dates=['first_active_month'])
dataset_overview(rTest)
rHistTransc['authorized_flag'] = rHistTransc['authorized_flag'].replace({'Y':1, 'N':0})
rHistTransc['category_1'] = rHistTransc['category_1'].replace({'Y':1, 'N':0})
rNewTransc['authorized_flag'] = rNewTransc['authorized_flag'].replace({'Y':1, 'N':0})
rNewTransc['category_1'] = rNewTransc['category_1'].replace({'Y':1, 'N':0})
tHistTrain = rHistTransc.merge(rTrain, on='card_id', how='inner')
tNewTrain = rNewTransc.merge(rTrain, on='card_id', how='inner')

tHistTest = rHistTransc.merge(rTest, on='card_id', how='inner')
tNewTest = rNewTransc.merge(rTest, on='card_id', how='inner')
rTrain.columns
(tHistTrain['card_id'].nunique())/rTrain['card_id'].nunique()
(tNewTrain['card_id'].nunique())/rTrain['card_id'].nunique()
tHistTrain = reduce_mem_usage(tHistTrain)
tHistTest = reduce_mem_usage(tHistTest)
tNewTrain = reduce_mem_usage(tNewTrain)
tNewTest = reduce_mem_usage(tNewTest)
tNewTest[cols].columns
tNewTest.merge(rMerchant.drop(['merchant_category_id', 'subsector_id', 'category_1', 'c'], axis=1), on=['merchant_id'], how='left')

tNewTrain['elapsed_time'] = (tNewTrain['first_active_month'].max() - tNewTrain['first_active_month'].dt.date).dt.days
tHistTrain['elapsed_time'] = (tHistTrain['first_active_month'].max() - tHistTrain['first_active_month'].dt.date).dt.days
dataset_overview(rHistTransc)
historical_transactions = reduce_mem_usage(historical_transactions)
new_transactions = reduce_mem_usage(new_transactions)

agg_fun = {'authorized_flag': ['mean']}
auth_mean = historical_transactions.groupby(['card_id']).agg(agg_fun)
auth_mean.columns = ['_'.join(col).strip() for col in auth_mean.columns.values]
auth_mean.reset_index(inplace=True)

authorized_transactions = historical_transactions[historical_transactions['authorized_flag'] == 1]
historical_transactions = historical_transactions[historical_transactions['authorized_flag'] == 0]
