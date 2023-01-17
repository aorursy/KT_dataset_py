# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
df=pd.read_csv('../input/911csv/911.csv')
df.dtypes
import pandas as pd

df = pd.read_csv("../input/911csv/911.csv")
df.head()
df.zip.value_counts().head()
df.twp.value_counts().head()
df['title'].nunique
df['Reason']=df['title'].apply(lambda title:title.split(':')[0]) #I'm creating a new variable from the titles.
df.head()
df['Reason'].value_counts() #distribution of titles
import seaborn as sns

sns.countplot(x="Reason", data=df, palette="viridis");
df.timeStamp
df['timeStamp']=pd.to_datetime(df['timeStamp']) #convert a timeStamp variable to a datetime
time = df['timeStamp'].iloc[0]

time.hour
time=df['timeStamp'].iloc[0]

df['Hour']=df['timeStamp'].apply(lambda time:time.hour)

df['Month']=df['timeStamp'].apply(lambda time:time.month)

df['Day']=df['timeStamp'].apply(lambda time:time.dayofweek)



#we create new features from timestamp variable. separating the variable in day,month and hour 
df.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day']=df['Day'].map(dmap)
df.head()
sns.countplot(x='Day',data=df,hue='Reason',palette='viridis'); #reason on daily basis.
sns.countplot(x='Hour',data=df,hue='Reason',palette='viridis'); #reason on hourly basis.
sns.countplot(x='Month',data=df,hue='Reason',palette='viridis'); # where is the months of sep,oct,nov?. these observations are not in the dataset
Monthnew=df.groupby('Month').count()

Monthnew.head(10)
Monthnew['twp'].plot(); # count of calls per month.
sns.lmplot(x='Month',y='twp',data=Monthnew.reset_index());

#Now see if we can use seaborn's lmplot() to create a linear fit on the number of calls per month and we reset the index to a column.
df['Date']=df['timeStamp'].apply(lambda p:p.date()) #new feature at timestamp column
df.head()
import matplotlib.pyplot as plt
df.groupby('Date').count()['twp'].plot()

plt.tight_layout();

#count of accidents(twp=township) by date.
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()



#If the call reason is traffic accidents.
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()



#If the call reason is fire accidents.
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()

#If the call reason is ems(Emergency Medical Services)
dayHour = df.groupby(['Day','Hour']).count().Reason.unstack()

dayHour.head(10)

#count of calls per day and hour
f,ax = plt.subplots(figsize=(10, 10)) #shape

sns.heatmap(dayHour,cmap='viridis', fmt="d", linewidths = .5);
sns.clustermap(dayHour,cmap='viridis', annot=True, fmt="d", linewidths = .5,figsize=(15, 15));
dayMonth = df.groupby(by=['Day','Month']).count()['Reason'].unstack()

dayMonth.head(10)
f,ax = plt.subplots(figsize=(10, 10))

sns.heatmap(dayMonth,cmap='viridis', annot=True, fmt="d",linewidths = .5);
sns.clustermap(dayHour,cmap='viridis', annot=True, fmt="d", linewidths = .5,figsize=(15, 15));