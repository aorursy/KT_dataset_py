import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
DataFrame=pd.read_csv('/kaggle/input/montcoalert/911.csv')
DataFrame.info()
DataFrame.head(3) 

#First three rows of data
DataFrame.tail(2)

#Last two rows of data
DataFrame['zip'].value_counts().head()
DataFrame['twp'].value_counts().head(5)
DataFrame['title'].nunique()
DataFrame['Reason']=DataFrame['title'].apply(lambda title:title.split(':')[0])

##title.split(':')[0] divide each string by : and takes the first word
DataFrame['Reason'].value_counts()
sns.countplot(x='Reason',data=DataFrame, palette='viridis')
type(DataFrame['timeStamp'].iloc[0])
DataFrame['timeStamp'] = pd.to_datetime(DataFrame['timeStamp'])
DataFrame['Hour']=DataFrame['timeStamp'].apply(lambda time:time.hour)

DataFrame['Month']=DataFrame['timeStamp'].apply(lambda time:time.month)

DataFrame['Day of Week']=DataFrame['timeStamp'].apply(lambda time:time.dayofweek)
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

DataFrame['Day of Week'] = DataFrame['Day of Week'].map(dmap)
sns.countplot(x='Day of Week',data=DataFrame,hue='Reason',palette='viridis')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
sns.countplot(x='Month',data=DataFrame,hue='Reason',palette='viridis')



# To relocate the legend

plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
DataFrame['Date']=DataFrame['timeStamp'].apply(lambda t: t.date())
plt.figure(figsize=(20,12))

DataFrame.groupby('Date').count()['twp'].plot()

plt.tight_layout()


plt.figure(figsize=(20,12))

DataFrame[DataFrame['Reason']=='Traffic'].groupby('Date').count()['twp'].plot()

plt.title('Traffic')

plt.tight_layout()
plt.figure(figsize=(20,12))

DataFrame[DataFrame['Reason']=='Fire'].groupby('Date').count()['twp'].plot()

plt.title('Fire')

plt.tight_layout()
plt.figure(figsize=(20,12))

DataFrame[DataFrame['Reason']=='EMS'].groupby('Date').count()['twp'].plot()

plt.title('EMS')

plt.tight_layout()
dayHour = DataFrame.groupby(by=['Day of Week','Hour']).count()['Reason'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis',

    col_colors=None,

    mask=None)