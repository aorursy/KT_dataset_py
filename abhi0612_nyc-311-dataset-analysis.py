import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
#importing the dataset

dataset=pd.read_csv("../input/311_Service_Requests_from_2010_to_Present.csv")

pd.options.display.max_columns = None

dataset.head(4)
dataset.info()
dataset.dtypes
dataset['Created Date']=pd.to_datetime(dataset['Created Date'])

dataset['Closed Date']=pd.to_datetime(dataset['Closed Date'])
dataset.head(5)
dataset['Request_Closing_Time'] = dataset['Closed Date']-dataset['Created Date']
dataset.head(5)
dataset.columns
dataset.drop(['Incident Address', 'Street Name', 'Cross Street 1', 'Cross Street 2',

       'Intersection Street 1', 'Intersection Street 2','Resolution Description', 

     'Resolution Action Updated Date','Community Board','X Coordinate (State Plane)','School or Citywide Complaint',

    'Vehicle Type','Taxi Company Borough','Taxi Pick Up Location','Garage Lot Name','School Name', 'School Number', 

              'School Region', 'School Code','School Phone Number', 'School Address', 'School City', 'School State',

       'School Zip', 'School Not Found','Ferry Direction', 'Ferry Terminal Name','Unique Key','Bridge Highway Name',

       'Bridge Highway Direction', 'Road Ramp', 'Bridge Highway Segment'],axis=1,inplace=True)
dataset.head()
dataset.info()
dataset.isna().sum()
#dropping created and closed date

dataset.drop(['Closed Date','Created Date'],axis=1,inplace=True)

dataset.head(5)
#dealing with missing values

dataset.isna().sum()
dataset['Agency'].value_counts()

dataset['Agency Name'].value_counts()

sns.countplot(dataset['Agency Name'])
dataset['Complaint Type'].value_counts().head()

plot=sns.countplot(dataset['Complaint Type'])

plot.set_xticklabels(plot.get_xticklabels(),rotation=90)
dataset['Descriptor'].isna().sum()
dataset['Descriptor'].describe()
dataset['Descriptor'].value_counts().head(5)
plot2=sns.countplot(dataset['Descriptor'])

plot2.set_xticklabels(plot.get_xticklabels(),rotation=90)
dataset['Location Type'].isna().sum()
dataset['Location Type'].value_counts().head()
dataset['Location Type'].fillna(value='Street/Sidewalk',inplace =True)

plot3=sns.countplot(dataset['Location Type'])

plot3.set_xticklabels(plot3.get_xticklabels(),rotation=90)
dataset['Incident Zip'].value_counts().head()

dataset['Incident Zip'].isna().sum()

dataset['Incident Zip'].fillna(value=11385,inplace=True)
dataset['Address Type'].value_counts()

dataset['Address Type'].fillna(value='Address',inplace=True)

sns.countplot(dataset['Address Type'])
dataset.drop(['Latitude', 'Longitude','Location','Y Coordinate (State Plane)','Landmark'],axis=1,inplace=True)
dataset.isna().sum()
dataset['City'].value_counts().head()

dataset['Facility Type'].value_counts().head()
dataset['City'].fillna(value='BROOKLYN',inplace=True)

dataset['City'].value_counts().head()

plot4=sns.countplot(x=dataset['City'])

plot4.set_xticklabels(plot.get_xticklabels(),rotation=90)
dataset.isna().sum()
dataset['Request_Closing_Time'].head()
dataset['Request_Closing_Time'].fillna(value=dataset['Request_Closing_Time'].mean(),inplace=True)
dataset['Request_Closing_Time'].isna().sum()
dataset['Request_Closing_Time'].dtypes
dataset.head(10)
dataset['Status'].value_counts()
sns.countplot(dataset['Status'])
#bivariate analysis

#the most common complaint

dataset['Complaint Type'].value_counts().head(6)
dataset.head(3)
desc=dataset.groupby(by='Complaint Type')['Descriptor'].agg('count')

desc
#City with their status

dataset.loc[dataset['City']=='NEW YORK',]['Borough'].value_counts()
#Newyork city has how many boroughs and whats their status 

sns.countplot(x=dataset.loc[dataset['City']=='NEW YORK',]['Borough'],hue='Status',data=dataset)
#Newyork city has max complaints of which complaint type?

dataset.loc[dataset['City']=='NEW YORK',:]['Complaint Type'].value_counts()
#Countplot to show Newyork city has max complaints of which complaint type?

plot=sns.countplot(x=dataset.loc[dataset['City']=='NEW YORK',:]['Complaint Type'])

plot.set_xticklabels(plot.get_xticklabels(),rotation = 90)
#Avg time taken to solve a case in Newyork city

dataset.loc[(dataset['City']=='NEW YORK')&(dataset['Status']=='Closed'),:]['Request_Closing_Time'].mean()
dataset.loc[(dataset['City']=='NEW YORK')&(dataset['Status']=='Closed'),:]['Request_Closing_Time'].std()
dataset['Borough'].value_counts()
dataset['Location Type'].value_counts()
#Top Location type and their countplot with hues='Borough'

sns.countplot(dataset.loc[dataset['Location Type'].isin(['Street/Sidewalk','Store/Commercial','Club/Bar/Restaurant'])]

              ['Location Type'],data=dataset,hue='Borough')

import datetime

dataset['year'] = pd.DatetimeIndex(dataset['Due Date']).year

dataset.head()
sns.countplot(dataset['year'],hue='Borough',data=dataset)
dataset['Location Type'].value_counts()
#Display the complaint type and city together

dataset[['Complaint Type','City']].head()
#Find the top 10 complaint types 

dataset['Complaint Type'].value_counts()[0:10,]
#Plot a bar graph of count vs. complaint types

plot3=sns.countplot(dataset['Complaint Type'])

plot3.set_xticklabels(plot3.get_xticklabels(),rotation =90)
#Display the major complaint types and their count

#top 5 complaint types

series=dataset['Complaint Type'].value_counts()[0:5,]

series.nlargest().index
#graph

plot4=sns.barplot(x=series.nlargest().index,y=series.nlargest().values)

plot4.set_xticklabels(plot3.get_xticklabels(),rotation =90)