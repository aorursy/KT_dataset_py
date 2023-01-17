# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import dask.dataframe as dd

import dask

import time

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
!ls -lsh ../input/PLAsTiCC-2018/test_set.csv
# monitor cpu memory usage

!free -g
%%time

# kernel died running the following pandas command due to OOM

#df = pd.read_csv('../input/PLAsTiCC-2018/test_set.csv') 



df = dd.read_csv('../input/PLAsTiCC-2018/test_set.csv') 
df
df.head() # dask just reads the head
%%time

df.shape # dask is not getting the actual shape since it is lazy
%%time

dask.compute(df.shape)
%%time

# simple column-wise reduction operations

df['flux'].mean().compute() # returns a scalar
!free -g
%%time

df_sample = df.loc[df.object_id==13].compute()
%%time



# it runs for more than 9 hours and is killed by kaggle.

#flux_stats_of_each_mjd = df.groupby('mjd').agg({'flux':['std']}).compute()

# This will return a pandas dataframe



#flux_stats_of_each_mjd.head()
#print(type(flux_stats_of_each_mjd),flux_stats_of_each_mjd.shape)