# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
# First, we need to measure the memory usage of our data 
def memory_usage(df):
    if isinstance (df, pd.DataFrame):                     # to see if we are dealing with dataframe 
        m_usage = df.memory_usage(deep = True).sum()
    else:                                                 # if it's not a dataframe, we assume it as a series
        m_usage = df.memory_usage(deep = True)
    
    m_usage_mb = m_usage / (1024**2)                      # m_usage is in bits and we would like something in megabytes
    return m_usage_mb
# Next we gain some insights on the training data
train = pd.read_csv("../input/train.csv")
print(train.info())

# Here we can see that, the training data takes 864.3 MB in total, which might take a while for processing
# All the numerical columns are either in int64 or float64, which takes 8 bytes
# Lets take a further look at int64
print(np.iinfo('int64'))

# Apart from the Ids, we don't expect other numerical columns to be larger than, say, 100000. Who the hell will have 100000 kills? 
# It is unnecessary to store our data in this form, fortunatelly pandas provides to_numerical method to fir the data into the most suffcient form
# We process integer columns and float columns saperately, if anyone comes up with an easier idea, feel free to replace mine! :)
train_int = train.select_dtypes(include = ['int64'])
train_float = train.select_dtypes(include = ['float64'])

print("Int columns memory use: ", round(memory_usage(train_int),2), "MB")
print("Float columns memory use: ", round(memory_usage(train_float),2), "MB")

# Now let's see what happens after we convert our columns into the right type
train_int_converted = train_int.apply(pd.to_numeric, downcast = 'unsigned')
train_float_converted = train_float.apply(pd.to_numeric, downcast = 'float')

print("Int columns memory use after conversion: ", round(memory_usage(train_int_converted),2), "MB")
print("Float columns memory use after concersion: ", round(memory_usage(train_float_converted),2), "MB")

# We saw a 80% memory usage decrease for integer columns and 50% for float columns, and a roughly 75% decrease in memory usage in total!
# To show the result in more details, let's see what happened

print(pd.concat([train_int_converted, train_float_converted], axis = 1, join = 'inner').info())

# All float columns are now in float 32 and most integer columns are down to int8, which makes sense as we do not expect crazy number of kills, headshots, healing etc.
# We can perform the same conversion for test data as well. Using the converted dataframes, the preprocessing and modelling process will be much more efficient :)

# This is the first time that I write a kernal, if you have any suggestions, please feel free to comment below.
# Hope this kernal help with your hacking :) Thank you!