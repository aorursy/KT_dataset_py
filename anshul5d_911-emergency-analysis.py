# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/911.csv')
df.columns
df.isnull().sum()
columns = {'lat':'Latitude','lng':'Longitude','desc':'Description of Emergency','zip':'ZIP Code','title':'Title'
,'timeStamp':'Datetime','twp':'Town','addr':'Adress'}
df = df.rename(columns,axis=1)
df.info()
#top 5 zip code
zipcode = df['ZIP Code'].value_counts()
zipcode.head()
#top 5 Township
df['Town'].value_counts().head()
# unique values of title columns
len(df['Title'].unique())
df['Title'].nunique()
#for getting the particular value like EMS,Fire 
x=df['Title'].iloc[0]
x
x.split(':')[0]
df['Reason'] = df['Title'].apply(lambda Title : Title.split(':')[0])
df['Reason'].value_counts()
sns.countplot(x='Reason',data = df)
df.info()
# type of Data and time of the call
type(df['Datetime'].iloc[0])
# date of column
df['Datetime']=pd.to_datetime(df['Datetime'])
df.shape
time = df['Datetime'].iloc[0]
time.hour
time.month
time.day
time.second
time.minute
df['Hour']= df['Datetime'].apply(lambda Datetime : Datetime.hour)
df['Hour'].head()
type(df['Datetime'].iloc[0].hour)
df['Month']=df['Datetime'].apply(lambda Datetime : Datetime.month)
df['DatOFWeek']=df['Datetime'].apply(lambda Datetime : Datetime.dayofweek)
dmap ={0:'Sun',1:'Mon',2:'Tue',3:'Wed',4:'Thu',5:'Fri',6:'Sat'}
df['DatOFWeek'] = df['DatOFWeek'].map(dmap)
df['DatOFWeek'].head()
sns.countplot(x='DatOFWeek', data = df,hue='Reason', palette='viridis').grid()
plt.legend(bbox_to_anchor =(1.05,1),loc=2,borderaxespad= 0.)
sns.countplot(x='Month', data= df,hue='Reason', palette='viridis').grid()
plt.legend(bbox_to_anchor = (1.05,1),loc=2)
ass=df['Month'].value_counts()
ass
byMonth = df.groupby('Month').count()
sns.lmplot(x='Month',y='Town', data= byMonth.reset_index())
t = df['Datetime'].iloc[0]
t
df['Date'] = df['Datetime'].apply(lambda t:t.date())
df.head()
df.groupby('Date').count()['Latitude'].plot()
plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date')['Latitude'].count().plot()
df[df['Reason']=='Fire'].groupby('Date')['Latitude'].count().plot().grid()
df[df['Reason']=='EMS'].groupby('Date')['Latitude'].count().plot().grid()
dayHour=df.groupby(['DatOFWeek','Hour']).count()['Reason'].unstack()
plt.figure(figsize=(12,6))
sns.heatmap(dayHour, cmap='viridis')
sns.clustermap(dayHour,cmap='coolwarm')
dayMonth = df.groupby(['DatOFWeek', 'Month']).count()['Reason'].unstack()
dayMonth
sns.heatmap(dayMonth,cmap = 'coolwarm')
sns.clustermap(dayMonth, cmap = 'coolwarm')
df['e'].value_counts()
df.shape
