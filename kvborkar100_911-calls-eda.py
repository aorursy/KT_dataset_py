# Import statements
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import seaborn as sns
dataset = pd.read_csv('../input/911.csv')
dataset.head()
dataset.info()
dataset.apply(lambda x:x.nunique())
dataset.title.value_counts()
dataset['type'] = dataset.title.apply(lambda x: x.split(':')[0])
dataset.type.value_counts()
plt.figure(figsize=(7,4),dpi=100)
sns.countplot(x=dataset.type)
plt.title("Call Distribution by type")
type(dataset.timeStamp[0])
dataset.timeStamp = pd.to_datetime(dataset.timeStamp)
#create three new columns for Month, Hour, Day
dataset['Month'] = dataset.timeStamp.apply(lambda x:x.month)
dataset['Hour'] = dataset.timeStamp.apply(lambda x:x.hour)
dataset['Day'] = dataset.timeStamp.apply(lambda x:x.dayofweek)
dataset.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
dataset['Day'] = dataset['Day'].map(dmap)
plt.figure(figsize=(7,4),dpi=100)
sns.countplot(x=dataset.Day)
plt.title('No of Calls per dayofweek')
# No of calls per Day of Week by Reason 
plt.figure(figsize=(7,4),dpi=100)
sns.countplot(x=dataset.Day,hue=dataset.type)
plt.title('No of Calls per dayofweek')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
# No of calls by Month
plt.figure(figsize=(7,4),dpi=100)
sns.countplot(x=dataset.Month,hue=dataset.type)
plt.title('No of Calls by Month')
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
byMonth = dataset.groupby('Month').count()
byMonth
# Plotting Calls per month using Line Plot
plt.figure(figsize=(7,4),dpi=100)
plt.title('no of calls per month')
byMonth.lat.plot()
g = sns.lmplot(x='Month',y='lat',markers='x',data=byMonth.reset_index(),size=8)
plt.title('linear fit to the no. of calls per Month')
dataset['Date'] = dataset.timeStamp.apply(lambda x:x.date())
dataset.Date.value_counts()
plt.figure(figsize=(7,4),dpi=100)
dataset.groupby('Date').count().lat.plot()
plt.title('No. of calls by Date')
plt.tight_layout()
plt.figure(figsize=(7,4),dpi=100)
dataset[dataset['type'] == 'Traffic'].groupby('Date').count()['lat'].plot()
plt.title('Reason -> Traffic (by Dates)')
plt.tight_layout()
plt.figure(figsize=(7,4),dpi=100)
dataset[dataset['type'] == 'Fire'].groupby('Date').count()['lat'].plot()
plt.title('Reason -> Fire (by Dates)')
plt.tight_layout()
plt.figure(figsize=(7,4),dpi=100)
dataset[dataset['type'] == 'EMS'].groupby('Date').count()['lat'].plot()
plt.title('Reason -> EMS (by Dates)')
plt.tight_layout()
dayHour = dataset.groupby(by=['Day','Hour']).count()['type'].unstack()
dayHour
plt.figure(figsize=(12,6))
sns.heatmap(dayHour)
dayMonth = dataset.groupby(by=['Day','Month']).count()['type'].unstack()
dayMonth
plt.figure(figsize=(12,6))
sns.heatmap(dayMonth)
