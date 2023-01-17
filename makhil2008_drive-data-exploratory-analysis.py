import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import glob

import seaborn as sns

import plotly.graph_objs as go

import plotly.plotly as py

sns.set(style="darkgrid")

import matplotlib.pyplot as plt

%matplotlib inline

import os


path = r'../input/input_data/drive' # use your path

all_files = glob.glob(path + "/*.parquet")



li = []



for filename in all_files:

    df = pd.read_parquet(filename, engine='auto')

    li.append(df)



driveframe = pd.concat(li, axis=0, ignore_index=True)
driveframe.describe().T
"Number of Unique Vechiles: " + str(len(driveframe.vehicle_id.drop_duplicates()))
"Number of Unique trips: " + str(len(driveframe.trip_id.drop_duplicates()))
# Checking the lvel of the data

driveframe.shape, driveframe[['vehicle_id','trip_id','datetime']].drop_duplicates().shape
# Max Time interval beteen two consicutive data points

"Max time difference between 2 data points of a trip: "+ str(((driveframe.groupby(['vehicle_id','trip_id']).datetime.shift(-1)-driveframe.datetime).fillna(0)/ np.timedelta64(1, 's')).max())+"s"
N_trips=driveframe.groupby('vehicle_id').trip_id.nunique().reset_index().sort_values('trip_id',ascending=False) 

plt.figure(figsize=(16, 6))

ax=sns.barplot(x=N_trips.trip_id, y=N_trips.vehicle_id, data=N_trips,orient='h',color='darkcyan',order=N_trips['vehicle_id'])

ax.set_xlabel('Number of Trips')
TripTime=driveframe.groupby('trip_id').agg({'datetime':['min','max']}).reset_index()

TripTime.columns=['trip_id','trip_start','trip_end']

TripTime['trip_time']=(TripTime['trip_end']-TripTime['trip_start']) / (np.timedelta64(1, 's')*60)

plt.figure(figsize=(13, 4))

ax=sns.distplot(TripTime['trip_time'])

ax.set_xlabel('Trip Time (in Minutes)')
plt.figure(figsize=(13, 4))

sns.distplot(driveframe['velocity'])