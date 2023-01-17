import warnings

warnings.filterwarnings(action="ignore")
import dask.dataframe as dd

import dask.array as da

import pandas as pd

import numpy as np



dask_df = dd.read_csv('/kaggle/input/us-accidents/US_Accidents_June20.csv')
# Checking dataframe partitions

dask_df.map_partitions(type).compute()
# Checking length of each partition

dask_df.map_partitions(len).compute()
#Accessing one of the partition

dask_df.partitions[1].compute()
dask_df.head()
dask_df.tail()
dask_df.dtypes
dask_df.describe().compute()
dask_df = dask_df.set_index('ID')

dask_df
dask_df['long_delay'] = dask_df['Severity']==4

dask_df.head()
dask_df['long_delay'].dtype                                  # output: bool

dask_df['long_delay'] = dask_df['long_delay'].astype('int')

dask_df['long_delay'].dtype                                  # output: int64
dask_df['Severity'].unique().compute()



'''

Output: 

0 3

1 2

2 4

3 1

'''
dask_df['Severity'].value_counts().compute()



'''

Output: 

2   2373210

3   998913

4   112320

1   29174

'''
dask_df.loc['A-3'].compute()
dask_df.loc[:,'City'].compute()
dask_df.loc['A-100','State'].compute()



'''

Output:

ID

A-100   OH

Name:  State, dtype:  object

'''
dask_df.loc['A-5':'A-9']
dask_df[dask_df['Start_Lat']== 39.865147].compute()
dask_df[da.logical_and(dask_df['Start_Lng']==-86.779770,dask_df['Start_Lat']== 36.194839)].compute()
dask_df.isna().sum(axis=0).compute()
dask_df['Wind_Speed(mph)'].isnull().sum().compute()  # output 454609
dask_df['Wind_Speed(mph)'] = dask_df['Wind_Speed(mph)'].fillna(10)
dask_df['Wind_Speed(mph)'].isnull().sum().compute()  # output 0
dask_df = dask_df.dropna(subset=['Zipcode'])
print(any(dask_df.columns=='long_delay'))       # output True

dask_df = dask_df.drop('long_delay',axis=1)

print(any(dask_df.columns=='long_delay'))       # output False  
byState = dask_df.groupby('State')
byState['Temperature(F)'].mean().compute()