# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



# Import the numpy, pandas, matplotlib, seaborn packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt
#Importing & Reading the Data

df=pd.read_csv("../input/uber-request-data/Uber Request Data.csv")

df
#Correcting the data types

df['Request timestamp'] = pd.to_datetime(df['Request timestamp'])

df['Drop timestamp'] = pd.to_datetime(df['Drop timestamp'])

df.head()
# Removing unnecessary columns

df = df.drop(['Driver id'], axis = 1)
df.tail()
# How many unique pickup points are present in uberReq?

print(df['Pickup point'].unique())
# How many observations are present in uberReq?

df.shape
df.info()
# Inspecting the Null values , column-wise

df.isnull().sum(axis=0)
df[(df['Drop timestamp'].isnull())].groupby('Status').size()
print(len(df['Request id'].unique()))

print(len(df['Pickup point'].unique()))

print(len(df['Status'].unique()))
# Checking if there are any duplicate values

len(df[df.duplicated()].index)
# Univariate analysis on Status column 

status = pd.crosstab(index = df["Status"], columns="count")     

status.plot.bar()
#Univariate analysis on Pickup Point column 

pick_point = pd.crosstab(index = df["Pickup point"], columns="count")     

pick_point.plot.bar()
# grouping by Status and Pickup point.

df.groupby(['Status', 'Pickup point']).size()
# Visualizing the count of Status and Pickup point bivariate analysis

sns.countplot(x=df['Pickup point'],hue =df['Status'] ,data = df)
# Request and Drop hours

df['Request Hour'] = df['Request timestamp'].dt.hour
# Time Slots

df['Request Time Slot'] = 'Early Morning'

df.loc[df['Request Hour'].between(5,8, inclusive=True),'Request Time Slot'] = 'Morning'

df.loc[df['Request Hour'].between(9,12, inclusive=True),'Request Time Slot'] = 'Late Morning'

df.loc[df['Request Hour'].between(13,16, inclusive=True),'Request Time Slot'] = 'Noon'

df.loc[df['Request Hour'].between(17,21, inclusive=True),'Request Time Slot'] = 'Evening'

df.loc[df['Request Hour'].between(21,24, inclusive=True),'Request Time Slot'] = 'Night'
# As Demand can include trips completed, cancelled or no cars available, we will create a column with 1 as a value

df['Demand'] = 1
# As Supply can only be the trips completed, rest all are excluded, so we will create a column with 1 as a supply value trips completed and 0 otherwise.

df['Supply'] = 0

df.loc[(df['Status'] == 'Trip Completed'),'Supply'] = 1
# Demand Supply Gap can be defined as a difference between Demand and Supply

df['Gap'] = df['Demand'] - df['Supply']

df.loc[df['Gap']==0,'Gap'] = 'Trip Completed'

df.loc[df['Gap']==1,'Gap'] = 'Trip Not Completed'
# Removing unnecessary columns

df = df.drop(['Request Hour', 'Demand', 'Supply'], axis=1)
df.head()
# Plot to find the count of the three requests, according to the defined time slots

sns.countplot(x=df['Request Time Slot'],hue =df['Status'] ,data = df)
# Plot to find the count of the status, according to both pickup point and the time slot

pickup_df = pd.DataFrame(df.groupby(['Pickup point','Request Time Slot', 'Status'])['Request id'].count().unstack(fill_value=0))

pickup_df.plot.bar()
# Plot to count the number of requests that was completed and which was not

sns.countplot(x=df['Gap'], data = df)
## Plot to count the number of requests that was completed and which was not, against the time slot

gap_timeslot_df = pd.DataFrame(df.groupby(['Request Time Slot','Gap'])['Request id'].count().unstack(fill_value=0))

gap_timeslot_df.plot.bar()
# Plot to count the number of requests that was completed and which was not, against pickup point

gap_pickup_df = pd.DataFrame(df.groupby(['Pickup point','Gap'])['Request id'].count().unstack(fill_value=0))

gap_pickup_df.plot.bar()
# Plot to count the number of requests that was completed and which was not, for the final analysis

gap_main_df = pd.DataFrame(df.groupby(['Request Time Slot','Pickup point','Gap'])['Request id'].count().unstack(fill_value=0))

gap_main_df.plot.bar()