#importing data from CSV file into pandas dataframe



import numpy as np

import pandas as pd

import matplotlib as mpl

import matplotlib.pyplot as plt

 

df_raw = pd.read_csv('/kaggle/input/Uber Request Data.csv')

df_raw.head()
df_uber=df_raw.copy()



#addressing data quality issues and converting request and drop timestamp to datetime format

df_uber['Request timestamp']=pd.to_datetime(df_raw['Request timestamp'])

df_uber['Drop timestamp']=pd.to_datetime(df_raw['Drop timestamp'])



#replacing blanks in column name 

df_uber.columns = [col.replace(' ', '_') for col in df_uber.columns]



#drop unnecessary columns

df_uber=df_uber.drop(['Request_id','Driver_id','Drop_timestamp'], axis=1)

df_uber.head()
#checking unique values in status column

df_uber['Status'].unique()
#dividing the trips into 6 sessions based on dt.hour from Request Timestamp



session_labels=['Late Night','Early Morning','Late Morning','Afternoon','Evening','Night']

df_uber=df_uber.assign(session=pd.cut(df_uber.Request_timestamp.dt.hour,[-1,4,8,12,16,20,24],labels=session_labels))

df_uber.head()
# plotting share/frequency of all "Trip Status" over the day to identify problem areas

plt.style.use('ggplot')

colors = ["#CC2529", "#8E8D8D","#008000"]

df_uber.groupby(['session','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)

plt.title('Total Count of all Trip Statuses')

plt.xlabel('Sessions')

plt.ylabel('Total Count of Trip Status')

plt.show()
# Filtering out only "Cancelled"  trips

df_tripscancelled=df_uber[df_uber["Status"].str.contains('Cancelled')==True]

df_tripscancelled=df_tripscancelled.reset_index(drop=True)

df_tripscancelled.head()
# plotting share/frequency of all Cancelled trips over the day to identify problem areas

plt.style.use('ggplot')

colors = ["#20B2AA", "#9400D3"]

df_tripscancelled.groupby(['session','Pickup_point']).Pickup_point.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)

plt.title('Count and Distribution of all "Cancelled" Trips over the day')

plt.xlabel('Sessions')

plt.ylabel('Total Count of "Cancelled" Trips')

plt.show()
# Filtering out only "Cancelled"  trips

df_nocar=df_uber[df_uber["Status"].str.contains('No Car')==True]

df_nocar=df_nocar.reset_index(drop=True)

df_nocar.head()
plt.style.use('ggplot')

colors = ["#20B2AA", "#9400D3"]

df_nocar.groupby(['session','Pickup_point']).Pickup_point.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)

plt.title('Count and Distribution of all "No Car Available" Trips over the day')

plt.xlabel('Sessions')

plt.ylabel('Total Count of "No Car Availble" Trips')

plt.show()
# Filtering out trips in the City to Airport route. Pick-up Point - City

df_citytoairport=df_uber[df_uber["Pickup_point"].str.contains('City')==True]

plt.style.use('ggplot')

colors = ["#CC2529", "#8E8D8D","#008000"]

df_citytoairport.groupby(['session','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)

plt.title('Total count of all Trip Statuses over the day for City to Airport route')

plt.xlabel('Sessions')

plt.ylabel('Total Count of Trips')

plt.show()
# Filtering out trips in the Airport to route. Pick-up Point - Airport

df_airporttocity=df_uber[df_uber["Pickup_point"].str.contains('Airport')==True]

plt.style.use('ggplot')

colors = ["#CC2529", "#8E8D8D","#008000"]

df_airporttocity.groupby(['session','Status']).Status.count().unstack().plot.bar(legend=True, figsize=(15,10), color=colors)

plt.title('Total count of all Trip Statuses over the day in the Airport to City route')

plt.xlabel('Sessions')

plt.ylabel('Total Count of Trips')

plt.show()
df_uber['supply_demand'] = ['Supply' if x == 'Trip Completed' else 'Demand' for x in df_uber['Status']]

df_uber.head()
#Plotting Supply and Demand on the City to Airport Route

df_citytoairport_supplydemand=df_uber[df_uber["Pickup_point"].str.contains('City')==True]

plt.style.use('ggplot')

df_citytoairport_supplydemand.groupby(['session','supply_demand']).supply_demand.count().unstack().plot.line(legend=True, figsize=(15,10))

plt.title('Supply-Demand curve for City to Airport Route')

plt.xlabel('Sessions')

plt.ylabel('Supply/Demand')

plt.show()
#Plotting Supply and Demand on the Airport to City route

df_airporttocity_supplydemand=df_uber[df_uber["Pickup_point"].str.contains('Airport')==True]

plt.style.use('ggplot')

df_airporttocity_supplydemand.groupby(['session','supply_demand']).supply_demand.count().unstack().plot.line(legend=True, figsize=(15,10))

plt.title('Supply-Demand curve for Airport to City Route')

plt.xlabel('Sessions')

plt.ylabel('Supply/Demand')

plt.show()