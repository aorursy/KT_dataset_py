import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

sns.set(style="white", color_codes=True)

import matplotlib.pyplot as plt
calls = pd.read_csv('../input/montcoalert/911.csv')
calls.head(3)
# top zip codes for 911 calls

calls['zip'].value_counts().head()
# top 5 townships for 911 calls 

calls['twp'].value_counts().head()
# unique titles

calls['title'].nunique()
# new colum "reason"

calls['reason'] = calls['title'].apply(lambda title : title.split(':')[0])
calls.reason.head()
#most common reason

calls['reason'].value_counts().head()
# plot 

sns.countplot(x='reason',data = calls)
calls.info()
calls['timeStamp'] = pd.to_datetime(calls['timeStamp'])

calls['timeStamp'][0].time()
calls['Hour'] = calls['timeStamp'].apply(lambda time : time.hour)

calls['Month'] = calls['timeStamp'].apply(lambda time : time.month)

calls['DayofWeek'] = calls['timeStamp'].apply(lambda time : time.dayofweek)

calls.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

calls['DayofWeek'] = calls['DayofWeek'].map(dmap)

calls.head(1)
sns.countplot(x='DayofWeek', data = calls,hue = 'reason')
sns.countplot(x='Month',data=calls,hue='reason')
byMonth = calls.groupby('Month').count()

byMonth.head()
sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
calls['Date'] = calls['timeStamp'].apply(lambda t : t.date())
calls.head()
calls.groupby('Date').count()['lat'].plot()

plt.tight_layout
calls[calls['reason'] == 'EMS'].groupby('Date').count()['lat'].plot()

plt.title('EMS')

plt.tight_layout
calls[calls['reason'] == 'Fire'].groupby('Date').count()['lat'].plot()

plt.title('Fire')

plt.tight_layout
dayHour = calls.groupby(by=['DayofWeek','Hour']).count()['reason'].unstack()

dayHour.head()
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis')
sns.clustermap(dayHour,cmap='viridis')