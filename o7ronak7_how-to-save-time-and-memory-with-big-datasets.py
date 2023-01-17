import pandas as pd #Analysis 
import matplotlib.pyplot as plt #Visulization
import seaborn as sns #Visulization
import numpy as np #Analysis 
from scipy.stats import norm #Analysis 
from sklearn.preprocessing import StandardScaler #Analysis 
from scipy import stats #Analysis 
import warnings 
warnings.filterwarnings('ignore')
%matplotlib inline
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.figure_factory as ff
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import datetime
import os
import time
import gc
%matplotlib inline
df_train = pd.read_csv('../input/train.csv')
df_test  = pd.read_csv('../input/test.csv')

df_train_sample = df_train.copy()
del df_train_sample
gc.collect()
df_train.head()
df_train.tail()
df_train.shape
dtypes = {
        'Id'                : 'uint32',
        'groupId'           : 'uint32',
        'matchId'           : 'uint16',
        'assists'           : 'uint8',
        'boosts'            : 'uint8',
        'damageDealt'       : 'float16',
        'DBNOs'             : 'uint8',
        'headshotKills'     : 'uint8', 
        'heals'             : 'uint8',    
        'killPlace'         : 'uint8',    
        'killPoints'        : 'uint8',    
        'kills'             : 'uint8',    
        'killStreaks'       : 'uint8',    
        'longestKill'       : 'float16',    
        'maxPlace'          : 'uint8',    
        'numGroups'         : 'uint8',    
        'revives'           : 'uint8',    
        'rideDistance'      : 'float16',    
        'roadKills'         : 'uint8',    
        'swimDistance'      : 'float16',    
        'teamKills'         : 'uint8',    
        'vehicleDestroys'   : 'uint8',    
        'walkDistance'      : 'float16',    
        'weaponsAcquired'   : 'uint8',    
        'winPoints'         : 'uint8', 
        'winPlacePerc'      : 'float16' 
}
train_dtypes = pd.read_csv('../input/train.csv', dtype=dtypes)
df_train = pd.read_csv('../input/train.csv')

#check datatypes:
train_dtypes.info()
#check datatypes:
df_train.info()
# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
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

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df_train = reduce_mem_usage(df_train)
df_train.info()
train_dtypes = pd.read_csv('../input/train.csv',nrows=10000 , dtype=dtypes)
train_dtypes.head()
train = pd.read_csv('../input/train.csv', skiprows=range(1, 3000000), nrows=10000, dtype=dtypes)
train.head()
del train; del train_dtypes;
gc.collect()
columns = ['Id', 'groupId', 'matchId','killPlace','killPoints','kills','killStreaks','longestKill','winPlacePerc']

dtypes = {
        'Id'                : 'uint32',
        'groupId'           : 'uint32',
        'matchId'           : 'uint16',   
        'killPlace'         : 'uint8',    
        'killPoints'        : 'uint8',    
        'kills'             : 'uint8',    
        'killStreaks'       : 'uint8',    
        'longestKill'       : 'float16',    
        'winPlacePerc'      : 'float16' 
}
example = pd.read_csv('../input/train.csv', usecols=columns, dtype=dtypes)
example.head()
debug = True
if debug:
    df_train = pd.read_csv('../input/train.csv',nrows=10000 , dtype=dtypes)
    df_test  = pd.read_csv('../input/test.csv', dtype=dtypes)
else:
    df_train = pd.read_csv('../input/train.csv', dtype=dtypes)
    df_test  = pd.read_csv('../input/test.csv', dtype=dtypes)