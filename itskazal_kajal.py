# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns


df=pd.read_csv('../input/911-emergency-call-visualization/911.csv')
   #creating dataframe
df.info()
df.head() #describing head of df
df.describe()
df['zip'].value_counts().head(5)  #top 5 head value
df['twp'].value_counts().head(5)
df['Reason']=df['title'].apply(lambda title: title.split(':')[0])    # using lambda for creating new column
df
df['title'].nunique()  #unique title code value
df['Reason'].value_counts()
sns.countplot(x='Reason', data=df,palette='Set2')
df['timeStamp']=pd.to_datetime(df['timeStamp'])
df
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)   #converting timestamp column from strings  to datetime object
df['Month'] = df['timeStamp'].apply(lambda time: time.month)
df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}     #assigning name to day of the week using dictionary
df['Day of Week']=df['Day of Week'].map(dmap)  
df
sns.countplot(x='Day of Week',data=df,hue='Reason', palette='rocket')      #simple count plot
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.5)
sns.countplot(x='Month',data=df,hue='Reason',palette='rocket')
plt.legend(bbox_to_anchor=(1.2, 1), loc=2, borderaxespad=0.)
bymonth=df.groupby('Month').count()

bymonth.head()
plt.plot(bymonth['twp'])
sns.lmplot(x='Month',y='twp',data=bymonth.reset_index())
df['Date']=df['timeStamp'].apply(lambda t: t.date())
df.groupby('Date').count()['twp'].plot()
plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()
plt.title('Traffic')
plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()
plt.title('Fire')
plt.tight_layout()
df[df['Reason']=='EMS'].groupby('Date').count()['twp'].plot()
plt.title('EMS')
plt.tight_layout()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()
dayHour.head() 
plt.figure(figsize=(12,6))    #ploting heatmap
sns.heatmap(dayHour,cmap='rocket')
sns.clustermap(dayHour,cmap='rocket')
dayMonth = df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()
dayMonth.head()
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth,cmap='rocket')
sns.clustermap(dayMonth,cmap='rocket')

