# Import numpy and pandas



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)





# Import visualization libraries and set %matplotlib inline. 



import seaborn as sns

import matplotlib.pyplot as plt

sns.set_style('whitegrid')

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



df = pd.read_csv('../input/montcoalert/911.csv')
df.info()
df.head()
df.describe()
df['zip'].value_counts().head(5)
df['twp'].value_counts().head(5)
df['title'].nunique()
df['Reason']=df['title'].apply(lambda title: title.split(':')[0])
df.head()
df['Reason'].value_counts()
sns.countplot(x='Reason',data=df,palette='viridis')
type(df['timeStamp'].iloc[0])

df['timeStamp']= pd.to_datetime(df['timeStamp'])
df['Hour'] = df['timeStamp'].apply(lambda time: time.hour)

df['Month'] = df['timeStamp'].apply(lambda time: time.month)

df['Day of Week'] = df['timeStamp'].apply(lambda time: time.dayofweek)
df.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
df['Day of Week'] = df['Day of Week'].map(dmap)
df.head()
sns.countplot(x ="Day of Week",data =df,hue = 'Reason',palette='rainbow')



#To relocate the legend

plt.legend(bbox_to_anchor=(1.05,1),loc = 2 ,borderaxespad=0)
sns.countplot(x = "Month", data = df,hue='Reason',palette="Set1")



#To relocate the legend

plt.legend(bbox_to_anchor=(1.05,1), loc=2,borderaxespad=0.)

byMonth = df.groupby("Month").count()

byMonth.head()
#Could be any column

byMonth["twp"].plot()
sns.lmplot(x="Month",y= "twp" , data=byMonth.reset_index())
df['Date']=df['timeStamp'].apply(lambda t: t.date())
df.groupby('Date').count()['twp'].plot()

plt.tight_layout()
df[df['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
df[df['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
df[df["Reason"]=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
#dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack(





dayHour = df.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour, cmap='coolwarm')
sns.clustermap(dayHour,cmap='coolwarm')
dayMonth=df.groupby(by=['Day of Week','Month']).count()['Reason'].unstack()

dayMonth.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayMonth,cmap='viridis')