# -*- coding: utf-8 -*-
%matplotlib inline

# @author: ryanzjlib@gmail.com
# to plot a heatmap

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import matplotlib.pyplot as plt
# read data
events = pd.read_csv('../input/events.csv')
# store original columns names
colnames = events.columns 

# subset the data to get info about a single device
device_events = events.loc[events.device_id == 1186608308763918427, :]
device_events.info()
device_events.head()
# to parse the timestamps for this person
def process_datetime(dt):
    '''a simple function to parse string time into several components'''
    dt = datetime.strptime(dt, '%Y-%m-%d %H:%M:%S')
    return [dt.weekday(), dt.hour]  # you can modify here to get other time components

device_events = device_events.merge(device_events.timestamp.apply(lambda x: pd.Series(process_datetime(x))), left_index=True, right_index = True)
device_events.columns = colnames.tolist() + ['day_of_week', 'hour']
device_events.info()
device_events.head()
# count the hourly events groupped by day_of_week
df = device_events[['day_of_week', 'hour', 'device_id']].groupby(['day_of_week', 'hour']).count()
m = df.unstack()
m
# plot a heat map
plt.pcolor(m, cmap='Reds')
plt.colorbar()
plt.show()