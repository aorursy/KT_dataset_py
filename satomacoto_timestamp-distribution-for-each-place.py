# if timestamp represents minutes
%matplotlib inline

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

types = {'row_id': np.dtype(int),
         'x': np.dtype(float),
         'y' : np.dtype(float),
         'accuracy': np.dtype(int),
         'place_id': np.dtype(int) }

train = pd.read_csv('../input/train.csv', dtype=types, index_col=0)
#test = pd.read_csv('../input/test.csv', dtype=types, index_col=0)
def add_data(df):
    df['hour'] = df.time//60%24
    df['day'] = df.time//60//24
    df['weekday'] = df.time//60//24%7

add_data(train)
# hourly

place_hourly = train.groupby(['place_id', 'hour']).size().unstack()
place_hourly.head(5)
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,5))
place_hourly.ix[1000015801].plot.bar(ax=axes[0])
place_hourly.ix[1000025138].plot.bar(ax=axes[1])
place_hourly.ix[1000063498].plot.bar(ax=axes[2])
# weekday

place_weekly = train.groupby(['place_id', 'weekday']).size().unstack()
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8,5))
place_weekly.ix[1000015801].plot.bar(ax=axes[0])
place_weekly.ix[1000025138].plot.bar(ax=axes[1])
place_weekly.ix[1000063498].plot.bar(ax=axes[2])