# Data and Setup



import numpy as np 

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

sns.set_style('whitegrid')



#Reading in the CSV file



df = pd.read_csv('../input/911.csv')

# Checking the info of the 'df'



df.info()
# Checking the head of the 911 calls dataset and asking for the first 5 results



df.head(5)
# Top five Zip Codes for the 911 calls



df['zip'].value_counts().head(5)
# Top Five townships (twp) for the 911 calls



df['twp'].value_counts().head(5)
df['title'].iloc[0] # shows the first instance in the title column
df['Reasons'] = df['title'].apply(lambda title: title.split(':')[1])
df['Reasons'].value_counts().head(5)
df['Departments'] = df['title'].apply(lambda title: title.split(':')[0])
df['Departments'].value_counts()
sns.countplot(x='Departments',data=df,palette='coolwarm')

plt.tight_layout()
df['timeStamp'].iloc[0] # the timeStamp is a string
df['timeStamp'] = pd.to_datetime(df['timeStamp']) # converting timeStamp into a Datetime Object
time = df['timeStamp'].iloc[0] # extracting first entry of the timeStamp

time.hour # can grab specific attributes from a Datetime object by calling them - hours in the first entry in the timeStamp column
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'} # 'Day of Week' is an integer from 0-6 which need to be converted into actual days of week.
df['Day of Week'] = df['Day of Week'].map(dmap)
sns.countplot(x='Day of Week',data=df,hue='Departments',palette='coolwarm')

# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
plt.figure(figsize=(8,6))

sns.countplot(x='Month',data=df,hue='Departments',palette='coolwarm') # For month coloumn now



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = df.groupby('Month').count() # Groupby object called byMonth

byMonth.head()
# Could be any column

byMonth['twp'].plot()


sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reasons'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='coolwarm')