# Imports

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import seaborn as sns
import matplotlib.pyplot as plt
import math
#import plotly
import plotly.graph_objs as go
import plotly.plotly as py
import plotly.tools as tls
import plotly.figure_factory as ff

#plotly.__version__

# List of Datafiles
print(os.listdir("../input"))
pitstops = pd.read_csv('../input/pitStops.csv')
results = pd.read_csv('../input/results.csv')
races = pd.read_csv('../input/races.csv')
circuits = pd.read_csv('../input/circuits.csv', encoding='latin1')
drivers = pd.read_csv('../input/drivers.csv', encoding='latin1')

#identify yellow flag
laptimes = pd.read_csv('../input/lapTimes.csv')
laptimes.head()

# sort df
laptimes.sort_values(by = ['raceId', 'driverId', 'lap'], inplace=True)

laptimes.head()
#calculating the "totalmilli" and creating apropriate column for it in the df
laptimes['totalmilli'] = laptimes.groupby(['raceId', 'driverId'])['milliseconds'].transform(pd.Series.cumsum)
#check the results
laptimes.iloc[55:65]
# laptimes_try.groupby(['raceId', 'driverId'])['totalmilli'].last()
# laptimes_try.groupby(['raceId', 'driverId'])['milliseconds'].sum()
# Creating the copies to mearge:
laptimes_2 = laptimes[['raceId', 'lap', 'position', 'totalmilli']].copy()
laptimes_3 = laptimes[['raceId', 'lap', 'position', 'totalmilli']].copy()
# Adding and subtractin "1" to each position, so than we can merge the "correct" position with the one in front of it:
laptimes_2['position'] = laptimes_2['position'] + 1
laptimes_2.rename(columns={'position': "position_plus_1", 'totalmilli' : 'totalmilli_plus_1'}, inplace=True)

laptimes_3['position'] = laptimes_3['position'] -1
laptimes_3.rename(columns={'position': "position_min_1", 'totalmilli' : 'totalmilli_min_1'}, inplace=True)
# Mearging two dataframes:
merged = pd.merge(laptimes, laptimes_2, how = 'left', left_on=['raceId', 'lap', 'position'],
                  right_on=['raceId', 'lap', 'position_plus_1'])
# Mearging two dataframes:
merged = pd.merge(merged, laptimes_3, how = 'left', left_on=['raceId', 'lap', 'position'],
                  right_on=['raceId', 'lap', 'position_min_1'])
# Calculating how far each car behind/in front:
merged['to_in_front'] = merged['totalmilli'] - merged['totalmilli_plus_1']
merged['to_behind'] = merged['totalmilli_min_1'] - merged['totalmilli']
# 'to_previous' has to be >= 0:
print("positive:", merged[merged['to_in_front']>0].shape)
print("equal zero:", merged[merged['to_in_front']==0].shape)
print('less than zero', merged[merged['to_in_front']<0].shape)
# So there is one case when it is negative, which does not make sense
# This is it:
merged[merged['to_in_front']<0]
# To make sure that the issue is not because our manipulation, I look at the initial database "laptimes"
# This is the data for previous lap:
laptimes[(laptimes['raceId']==985)&(laptimes['lap']==55)].sort_values(by=['position'])
# This is tha data for the lap where we have a negative value
laptimes[(laptimes['raceId']==985)&(laptimes['lap']==56)].sort_values(by=['position'])
# Additional check. Just look at some random "lap". The 'totalmilli_plus_1' must be shifted dow.
merged[(merged['raceId']==984)&(merged['lap']==12)].sort_values(by=['position'])
# Now we can delete 'position_plus_1' and 'totalmilli_plus_1' columns if needed.

merged.drop(['position_plus_1', 'totalmilli_plus_1', 'position_min_1', 'totalmilli_min_1'], axis=1, inplace = True)
# Puting merged df into laptimes
laptimes = merged.copy()
laptimes.head()
