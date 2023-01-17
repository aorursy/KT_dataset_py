# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import matplotlib.pyplot as plt

%matplotlib inline
df=pd.read_csv('../input/911_calls_for_service.csv')

df.head()
df['priority'].value_counts()
import seaborn as sns

sns.countplot(x='priority',data=df,palette='viridis')
df['callDateTime'].iloc[0]
df['callDateTime']=pd.to_datetime(df['callDateTime'])
time=df['callDateTime'].iloc[0]

df['Hour']=df['callDateTime'].apply(lambda time:time.hour)

df['Month']=df['callDateTime'].apply(lambda time:time.month)

df['Day of Week']=df['callDateTime'].apply(lambda time:time.dayofweek)
dmap={0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

df['Day of Week']=df['Day of Week'].map(dmap)
sns.countplot(x='Day of Week',data=df,hue='priority',palette='viridis')
sns.countplot(x='Month',data=df,hue='priority',palette='viridis')
byMonth=df.groupby('Month').count()

byMonth.head()
byMonth['incidentLocation'].plot()
sns.lmplot(x='Month',y='incidentLocation',data=byMonth.reset_index())
df['Date']=df['callDateTime'].apply(lambda p:p.date())
df.groupby('Date').count()['incidentLocation'].plot()

plt.tight_layout()
df[df['priority']=='Medium'].groupby('Date').count()['incidentLocation'].plot()

plt.title('Medium')

plt.tight_layout()
df[df['priority']=='Low'].groupby('Date').count()['incidentLocation'].plot()

plt.title('Low')

plt.tight_layout()
df[df['priority']=='High'].groupby('Date').count()['incidentLocation'].plot()

plt.title('High')

plt.tight_layout()
df[df['priority']=='Non-Emergency'].groupby('Date').count()['incidentLocation'].plot()

plt.title('Non-Emergency')

plt.tight_layout()
df[df['priority']=='Emergency'].groupby('Date').count()['incidentLocation'].plot()

plt.title('Emergency')

plt.tight_layout()
df[df['priority']=='Out of Service'].groupby('Date').count()['incidentLocation'].plot()

plt.title('Out of Service')

plt.tight_layout()
dayHour = df.groupby(by=['Day of Week','Hour']).count()['priority'].unstack()

dayHour.head()
sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')
dayMonth = df.groupby(by=['Day of Week','Month']).count()['priority'].unstack()

dayMonth.head()
sns.heatmap(dayMonth,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')