# MATHEMATICS
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import random
import scipy

# SYSTEM
import os 
import gc

#VIZUALISATION
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
#from ipywidgets import Layout
#import plotly.offline as py
#py.init_notebook_mode(connected=True)

#from IPython.core.display import display, HTML
#display(HTML("<style>.container { width:100% !important; }</style>"))

#w = catboost.CatboostIpythonWidget('')
#w.update_widget()

# set-up
Kaggle_kernel=False
local_path='input/'
kaggle_path='../input/'
original_features=['app','channel','ip','device','os']
def load_data(name,rows=None):
    ''' Load the csv files into a TimeSeries dataframe with minimal data types to reduce the used RAM space. 
    It also saves the files in parquet file to reduce loading time by a factor of ~10.

    Arg:
    
        -name (str): ante_day, last_day, train, train_sample or test

    Returns:
        pd.DataFrame, with int index equal to 'click_id'
    '''

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
    else:
        types['is_attributed']='bool'

    # Defining csv file reading parameters
    read_args={
        'nrows':rows,
        'parse_dates':['click_time'],
        'infer_datetime_format':True,
        'index_col':'click_time',
        'usecols':list(types.keys()),
        'dtype':types,
        'engine':'c',
        'sep':','
        }

    # Setting file path
    file_path='{}{}'.format(kaggle_path,name)

    with open('{}.csv'.format(file_path),'rb') as File:
        data=(pd
            .read_csv(File,**read_args)
            .tz_localize('UTC')
            .tz_convert('Asia/Shanghai')
            .reset_index()
        )

    # Sorting frames
    if name=='test': # making sure index == click_id
        data=data.sort_values(by=['click_id']).reset_index(drop=True)
    elif name=='train_sample': # sorting time randomized by sampling
        data=data.sort_values(by=['click_time']).reset_index(drop=True)

    return data

def actor(dataframe,maxima):
    result=pd.Series(data=0,index=dataframe.index,dtype=np.uint64)
    for i in original_features:
        result=(dataframe[i]+1)+(maxima[i]+1)*result
    return result
df=load_data('train')
maxima=df.describe().loc['max',:].astype(np.uint32)
df.info()
gc.collect()
df=df.assign(actor=lambda x: actor(x,maxima)).drop(original_features,axis=1)
df.head()
gc.collect()
duplicates=df.loc[df.duplicated(subset=['click_time','actor'],keep=False),:]
gc.collect()
print('The proportion of duplicated rows in the training data is : {0:.1f}%'.format(df.duplicated(subset=['click_time','actor'],keep=False).mean()*100))
print('The attribution rate of the duplicates is {:.3f}% while the full set is {:.3f}%:'.format(duplicates.is_attributed.mean()*100,df.is_attributed.mean()*100))
print('The attribution rate for first elements of duplicates is {:.3f}%'.format(duplicates.loc[duplicates.duplicated(subset=['click_time','actor'],keep='first'),'is_attributed'].mean()*100))
print('The attribution rate for last elements of duplicates is {:.3f}%'.format(duplicates.loc[duplicates.duplicated(['click_time','actor'],keep='last'),'is_attributed'].mean()*100))
duplicates_dist=(duplicates
                 .groupby(['click_time','actor'])
                 .is_attributed.agg(['count','mean'])
                 .groupby('count')
                 .agg(['count','mean'])
                )
duplicates_dist=(duplicates_dist.rename(columns={'mean':'training'},level=0)
                 .rename(columns={'mean':'avg attr rate'},level=1)
                 .rename_axis('')
                )
del(df,duplicates)
gc.collect()
duplicates_dist
df=load_data('test')
maxima=df.describe().loc['max',:].astype(np.uint32)
df.info()
gc.collect()
df=df.assign(actor=lambda x: actor(x,maxima)).drop(original_features,axis=1)
df.head()
gc.collect()
print('The proportion of duplicated rows is : {0:.1f}%'.format(df.duplicated(subset=['click_time','actor'],keep=False).mean()*100))
duplicates=df.loc[df.duplicated(subset=['click_time','actor'],keep=False),:]
print('The table below shows statistics by number of duplicates:')
duplicates_dist[('test','count')]=duplicates.groupby(['click_time','actor']).click_id.count().value_counts()
duplicates_dist[('test','count')]=duplicates_dist[('test','count')].fillna(0).astype(int)
duplicates_dist