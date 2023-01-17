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
%%time
# Reading File
train_path  = '../input/train.csv'

# Set columns to most suitable type to optimize for memory usage
traintypes = {'fare_amount': 'float32',
              'pickup_datetime': 'str', 
              'pickup_longitude': 'float32',
              'pickup_latitude': 'float32',
              'dropoff_longitude': 'float32',
              'dropoff_latitude': 'float32',
              'passenger_count': 'uint8'}

cols = list(traintypes.keys())

train_df = pd.read_csv(train_path, usecols=cols, dtype=traintypes)
train_df['pickup_datetime'] = train_df['pickup_datetime'].str.slice(0, 19)
train_df['pickup_datetime'] = pd.to_datetime(train_df['pickup_datetime'],format = "%Y-%m-%d %H:%M",utc=True)
# 2015
train_df[train_df.pickup_datetime.dt.year==2015].reset_index(drop=True).to_feather('nyc_taxi_2015.feather')
# 2014
train_df[train_df.pickup_datetime.dt.year==2014].reset_index(drop=True).to_feather('nyc_taxi_2014.feather')
# 2013
train_df[train_df.pickup_datetime.dt.year==2013].reset_index(drop=True).to_feather('nyc_taxi_2013.feather')
# 2012
train_df[train_df.pickup_datetime.dt.year==2012].reset_index(drop=True).to_feather('nyc_taxi_2012.feather')
# 2011
train_df[train_df.pickup_datetime.dt.year==2011].reset_index(drop=True).to_feather('nyc_taxi_2011.feather')
# 2010
train_df[train_df.pickup_datetime.dt.year==2010].reset_index(drop=True).to_feather('nyc_taxi_2010.feather')
# 2009
train_df[train_df.pickup_datetime.dt.year==2009].reset_index(drop=True).to_feather('nyc_taxi_2009.feather')

# The files can be found in outputs