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
import seaborn as sns

import matplotlib.pyplot as plt
call  = pd.read_csv('/kaggle/input/montcoalert/911.csv')
call.head()
call.info()
call['zip'].value_counts().head()
call['twp'].value_counts().head()
call['title'].nunique()
call['Reason'] = call['title'].apply(lambda x:x.split(':')[0])
call['Reason'].value_counts()
sns.countplot(x='Reason',data=call, palette = 'viridis')
type(call['timeStamp'].iloc[0])
call['timeStamp'] = pd.to_datetime(call['timeStamp'])
type(call['timeStamp'].iloc[0])
call['timeStamp']
time = call['timeStamp'].iloc[0]

time.hour

# time.dayofweek

# time.month

# time.date()
call['Hour'] = call['timeStamp'].apply(lambda time: time.hour)

call['Month'] = call['timeStamp'].apply(lambda time:time.month)

call['Day of Week'] = call['timeStamp'].apply(lambda time:time.dayofweek)

call.sample(5)
dmap={0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat', 6:'Sun'}
call['Day of Week'] = call['Day of Week'].map(dmap)

call.head()
sns.countplot(x='Day of Week',data=call, hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1.05, 1))
sns.countplot(x='Month',data=call, hue='Reason',palette='viridis')

plt.legend(bbox_to_anchor=(1, 1))
byMonth = call.groupby('Month').count()

byMonth
sns.pointplot(x='Month', y = 'lat', data=byMonth.reset_index(), markers='o')
t = call['timeStamp'].iloc[0]

t.date()
call['Date'] = call['timeStamp'].apply(lambda t: t.date())

call['Date']
call.sample(5)
call.groupby('Date').count().head()
call.groupby('Date').count()['lat'].plot()
call[call['Reason']=='EMS'].groupby(by='Date').count()['lat'].plot()

plt.title('EMS')
call[call['Reason']=='Fire'].groupby(by='Date').count()['lat'].plot()

plt.title('Fire')
call[call['Reason']=='Traffic'].groupby(by='Date').count()['lat'].plot()

plt.title('Traffic')
dayHour = call.groupby(by=['Day of Week', 'Hour']).count()['Reason'].unstack()

dayHour
plt.figure(figsize=(12,6))

sns.heatmap(dayHour)
dayMonth = call.groupby(by=['Day of Week', 'Month']).count()['Reason'].unstack()

dayMonth
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth)