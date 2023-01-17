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
df_trip = pd.read_csv('../input/trip.csv')
df_trip.head()
df_trip['duration'] = df_trip['duration'].apply(lambda x : x / 60.0)
df_agg = df_trip[((df_trip['start_station_id'] == df_trip['end_station_id']) & (df_trip['duration'] < 2))].groupby(['bike_id', 'start_station_id', 'end_station_id']).aggregate({'duration'  : ['sum']})
df_agg.plot()
