# import the numpy as pandas libraries

import pandas as pd

import numpy as np


# import thr visualisation libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
# import thr visualisation libraries

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline 
# Read the csv file as a dataframe

data = pd.read_csv('../input/montcoalert/911.csv')
# Check the head of the data

data.head()
# Check the info of the data

data.info()
# Check the top 10 zipcodes for 911 codes

data['zip'].value_counts().head(10)
# Top 5 townships for 911 calls.

data['twp'].value_counts().head(5)
# Number of unique codes in the title column

len(data['title'].unique())
# Creating new features 

data['Reason'] = data['title'].apply(lambda title: title.split(':')[0])

data['Reason']
# Most common Reason for a 911 call

data['Reason'].value_counts()
# Use seaborn to create a count plot of the 911 calls by Reason

sns.countplot(x='Reason', data=data, palette='viridis')
# Data type of the time information

type(data['timeStamp'].iloc[0])
# Convert timeStamps to DateTime objects

data['timeStamp'] = pd.to_datetime(data['timeStamp'])
data['timeStamp'].head(2)
# Grab time by unique attributes

data['Hour'] = data['timeStamp'].apply(lambda time: time.hour)

data['Month'] = data['timeStamp'].apply(lambda time: time.month)

data['Day of Week'] = data['timeStamp'].apply(lambda time: time.dayofweek)

data['Date'] = data['timeStamp'].apply(lambda time: time.date())

data.head(2)
# Map the day of the week with string

dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

data['Day of Week'] = data['Day of Week'].map(dmap)
data['Day of Week'].value_counts()
# Seaborn to create countplot of the day of the week with Reason as the hue

sns.countplot(x='Day of Week', data=data, hue='Reason',palette='viridis')

plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
# Count plot of the month with Reason as the hue

sns.countplot(x='Month', data=data,hue='Reason',palette='viridis')

plt.legend(loc='center right', bbox_to_anchor=(1.25, 0.5))
#  group the DataFrame by the month column and use the count() method for aggregation.

byMonth = data.groupby('Month').count()

byMonth.head(3)
# plot off of the dataframe indicating the count of calls per month.

byMonth['twp'].plot()
# lmplot of the data grouped by Month

sns.lmplot(x='Month', y='twp', data=byMonth.reset_index())
# Create a plot by date

plt.figure(figsize=(12,4))

data.groupby('Date')['twp'].count().plot()

plt.tight_layout()
# Plot the data representing EMS.

plt.figure(figsize=(12,5))

data[data['Reason'] == 'EMS'].groupby('Date')['twp'].count().plot()

plt.title('EMS')

plt.tight_layout()
data['Reason'].unique()
# Plot the data representing Fire.

plt.figure(figsize=(12,5))

data[data['Reason'] == 'Fire'].groupby('Date')['twp'].count().plot()

plt.title('Fire')

plt.tight_layout()
# Plot the data representing Traffic.

plt.figure(figsize=(12,5))

data[data['Reason'] == 'Traffic'].groupby('Date')['twp'].count().plot()

plt.title('Traffic')

plt.tight_layout()
dayHour = data.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()

dayHour.head(3)
# create a heatmap using the new dayHour dataframe

plt.figure(figsize=(12,7))

sns.heatmap(data=dayHour, cmap='viridis')
dayMonth =  data.groupby(by=['Day of Week', 'Month']).count()['Reason'].unstack()

dayMonth.head(3)
# create a heatmap by the dayMonth dataframe

plt.figure(figsize=(12,7))

sns.heatmap(data=dayMonth,annot=False, cmap='viridis')