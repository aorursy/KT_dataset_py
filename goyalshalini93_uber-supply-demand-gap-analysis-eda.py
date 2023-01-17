# Supress Warnings



import warnings

warnings.filterwarnings('ignore')



# Import the numpy, pandas, matplotlib, seaborn packages



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import datetime as dt
#Reading Uber Request data

uberReq = pd.read_csv('../input/Uber Request Data.csv',encoding = "ISO-8859-1")

uberReq.head()
#Correcting the data types

uberReq['Request timestamp'] = pd.to_datetime(uberReq['Request timestamp'])

uberReq['Drop timestamp'] = pd.to_datetime(uberReq['Drop timestamp'])

uberReq.head()
# Removing unnecessary columns

uberReq = uberReq.drop(['Driver id'], axis = 1)
uberReq.tail()
#How many unique pickup points are present in uberReq?

print(uberReq['Pickup point'].unique())
#How many observations are present in uberReq?

uberReq.shape
uberReq.info()
#Inspecting the Null values , column-wise

uberReq.isnull().sum(axis=0)
uberReq[(uberReq['Drop timestamp'].isnull())].groupby('Status').size()
print(len(uberReq['Request id'].unique()))

print(len(uberReq['Pickup point'].unique()))

print(len(uberReq['Status'].unique()))
#Checking if there are any duplicate values

len(uberReq[uberReq.duplicated()].index)
#Univariate analysis on Status column 

status = pd.crosstab(index = uberReq["Status"], columns="count")     

status.plot.bar()
#Univariate analysis on Pickup Point column 

pick_point = pd.crosstab(index = uberReq["Pickup point"], columns="count")     

pick_point.plot.bar()
#grouping by Status and Pickup point.

uberReq.groupby(['Status', 'Pickup point']).size()
# Visualizing the count of Status and Pickup point bivariate analysis

sns.countplot(x=uberReq['Pickup point'],hue =uberReq['Status'] ,data = uberReq)
#Request and Drop hours

uberReq['Request Hour'] = uberReq['Request timestamp'].dt.hour
#Time Slots

uberReq['Request Time Slot'] = 'Early Morning'

uberReq.loc[uberReq['Request Hour'].between(5,8, inclusive=True),'Request Time Slot'] = 'Morning'

uberReq.loc[uberReq['Request Hour'].between(9,12, inclusive=True),'Request Time Slot'] = 'Late Morning'

uberReq.loc[uberReq['Request Hour'].between(13,16, inclusive=True),'Request Time Slot'] = 'Noon'

uberReq.loc[uberReq['Request Hour'].between(17,21, inclusive=True),'Request Time Slot'] = 'Evening'

uberReq.loc[uberReq['Request Hour'].between(21,24, inclusive=True),'Request Time Slot'] = 'Night'
#As Demand can include trips completed, cancelled or no cars available, we will create a column with 1 as a value

uberReq['Demand'] = 1
#As Supply can only be the trips completed, rest all are excluded, so we will create a column with 1 as a supply value trips completed and 0 otherwise.

uberReq['Supply'] = 0

uberReq.loc[(uberReq['Status'] == 'Trip Completed'),'Supply'] = 1
#Demand Supply Gap can be defined as a difference between Demand and Supply

uberReq['Gap'] = uberReq['Demand'] - uberReq['Supply']

uberReq.loc[uberReq['Gap']==0,'Gap'] = 'Trip Completed'

uberReq.loc[uberReq['Gap']==1,'Gap'] = 'Trip Not Completed'
#Removing unnecessary columns

uberReq = uberReq.drop(['Request Hour', 'Demand', 'Supply'], axis=1)
uberReq.head()
# Plot to find the count of the three requests, according to the defined time slots

sns.countplot(x=uberReq['Request Time Slot'],hue =uberReq['Status'] ,data = uberReq)
# Plot to find the count of the status, according to both pickup point and the time slot

pickup_df = pd.DataFrame(uberReq.groupby(['Pickup point','Request Time Slot', 'Status'])['Request id'].count().unstack(fill_value=0))

pickup_df.plot.bar()
#Plot to count the number of requests that was completed and which was not

sns.countplot(x=uberReq['Gap'], data = uberReq)
##Plot to count the number of requests that was completed and which was not, against the time slot

gap_timeslot_df = pd.DataFrame(uberReq.groupby(['Request Time Slot','Gap'])['Request id'].count().unstack(fill_value=0))

gap_timeslot_df.plot.bar()
#Plot to count the number of requests that was completed and which was not, against pickup point

gap_pickup_df = pd.DataFrame(uberReq.groupby(['Pickup point','Gap'])['Request id'].count().unstack(fill_value=0))

gap_pickup_df.plot.bar()
#Plot to count the number of requests that was completed and which was not, for the final analysis

gap_main_df = pd.DataFrame(uberReq.groupby(['Request Time Slot','Pickup point','Gap'])['Request id'].count().unstack(fill_value=0))

gap_main_df.plot.bar()