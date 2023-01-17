# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

#for dirname, _, filenames in os.walk('/kaggle/input'):

#    for filename in filenames:

#        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#!pip install vaex

!pip install "dask[complete]"
import dask

import dask.dataframe as dd



# It is just a logical read , nothing in memory at this moment

df = dd.read_csv('../input/riiid-test-answer-prediction/train.csv',low_memory=False)
# As a logical read only column info is there not about rows 

df.info()
# Delayed for rows just to 

df.shape
# but you can read as you want 

# like simple dataframe

# find more on dask site https://stories.dask.org/en/latest/

df.head()
%%time

# Loop over 1,000,000 ( 1 million )

# look time taken

for index , data in df.iterrows():

    #print(data)

    #print(20*'--')

    if index == 1000*1000 :

        break

        
%%time



# Find correct answer of each user 

df.groupby(df.user_id).answered_correctly.sum().compute()
# From above output last user id is 2147482888     

# lets check tail of dataframe

df.tail()