#importing libraries

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns # for creating plots

sns.set_style('whitegrid')

# loading csv data to dataframe 

df = pd.read_csv('../input/911.csv')

# checking top 5 records

df.head(5)
#checking columns and total records

df.info()
#checking the top 5 zip which made highest calls to 911

df['zip'].value_counts().head(5)
#checking the top 5 townships(twp) for 911 calls

df['twp'].value_counts().head(5)
# Displaying unique Title codes count

#len(df['title'].unique())

df['title'].nunique()
# Displaying unique Titles

df['title'].unique()
# Title is the category of the call

#The table below shows us that Vehicle Accident is the most common reason of 911-calls

df['title'].value_counts()

#Creating new column and copying the Title Codes from title column by using - apply()

#Title Codes are : EMS, Fire, Traffic



"""df['title'].iloc[0] #ist row will print EMS: BACK PAINS/INJURY 

title = df['title'].iloc[0] 

title  # will print "EMS: BACK PAINS/INJURY"  and is a string  

title.split(':') # spliting the title values, splitting key is ':'

title.split(':')[0] #selecting the ist value from list - will be used to create the lambda expression"""



#using apply() with custom lambda expression to create a new column called

#Reason  that contains the string value

df['Reason'] = df['title'].apply(lambda title:title.split(':')[0])

df[['title','Reason']].head(5)

#Displaying most common Reason for a 911 call

df['Reason'].value_counts()
# Using Seaborn to create a countplot of 911 calls by Reason

sns.countplot(x='Reason', data=df, palette='Blues')
# 1. The 'timestamp' column are actually in string. Converting them to DateTime objects

# 2. Using .apply() to create 3 new columns called Hour, Month, and Day of Week. 



df['timeStamp'] =pd.to_datetime(df['timeStamp']) #converting to datatime

time = df['timeStamp'].iloc[0] # grabbing ist value

time.hour # extracting the hour 

time.month #extracting the month

time.dayofweek #extracting the day of week

# New column Hour creating

df['Hour'] = df['timeStamp'].apply(lambda time:time.hour)

df['Hour'].head(2)
df['Month'] = df['timeStamp'].apply(lambda time:time.month)

df[['Month','Hour']].head(2)
df['Day of Week'] = df['timeStamp'].apply(lambda time:time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df['Day of Week'].head(1)
df[['timeStamp','Month','Hour','Day of Week']].head(5)
"""df['Day of Week'] = df['timeStamp'].apply(lambda time:time.strftime("%A"))

df[['timeStamp','Month','Hour','Day of Week']].head(5)"""
# Using seaborn to create a countplot of the Day of Week column.

# with the hue based off of the Reason column. 

sns.countplot(x='Day of Week', data=df, hue='Reason', palette='Blues')

# placing legend outside of the plot

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

# Using seaborn to create a countplot of the Month column.

# with the hue based off of the Reason column. 

sns.countplot(x='Month', data=df, hue='Reason', palette='Blues')

plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")