import numpy as np

import pandas as pd

import matplotlib

import matplotlib.pyplot as plt

import seaborn as sns



# Print versions of libraries

print(f"Numpy version : Numpy {np.__version__}")

print(f"Pandas version : Pandas {pd.__version__}")

print(f"Matplotlib version : Matplotlib {matplotlib.__version__}")

print(f"Seaborn version : Seaborn {sns.__version__}")



# Magic Functions for In-Notebook Display

%matplotlib inline



# Setting seabon style

sns.set(style='darkgrid')
data = pd.read_csv('../input/montcoalert/911.csv', encoding='latin_1')
data.head(5)
data.shape
data.info()
data.columns
data.drop('e', axis=1, inplace=True)
type(data['timeStamp'].iloc[0])
data['timeStamp'] = pd.to_datetime(data['timeStamp'])
mindate = data["timeStamp"].min()

mindate
maxdate = data["timeStamp"].max()

maxdate
from dateutil import relativedelta



dif = relativedelta.relativedelta(pd.to_datetime(maxdate), pd.to_datetime(mindate))

print("{} years and {} months".format(dif.years, dif.months))
data['Hour'] = data['timeStamp'].apply(lambda time: time.hour)

data['Hour'].head()
data['DayOfWeek'] = data['timeStamp'].apply(lambda time: time.dayofweek)

data['DayOfWeek'].head()
data['Month'] = data['timeStamp'].apply(lambda time: time.month)

data['Month'].head()
data['Year'] = data['timeStamp'].apply(lambda time: time.year)

data['Year'].head()
data['Date'] = data['timeStamp'].apply(lambda time:time.date())

data['Date'].head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}
data['DayOfWeek'] = data['DayOfWeek'].map(dmap)

data['DayOfWeek'].head()
data.head(5)
total = data.isnull().sum().sort_values(ascending=False)

percent = ((data.isnull().sum()/data.isnull().count())*100).sort_values(ascending=False)

missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])

missing_data.head(20)
pd.set_option('display.max_colwidth', -1)

data['desc'].head()
data['station_code'] = data['desc'].str.split('Station', expand=True)[1].str.split(';', expand=True)[0]

data['station_code'] = data['station_code'].str.replace(':', '')

data['station_code'].head()
data['station_code'] = data['station_code'].str.strip()

data['station_code'].head()
data['reason_category'] = data['title'].apply(lambda title: title.split(':')[0])

data['reason_category'].head()
data['reason'] = data['title'].apply(lambda title: title.split(':')[1])

data['reason'].head()
data.head(5)
data['twp'].value_counts().head(10)
data[data['twp']=='LOWER MERION']['reason_category'].value_counts()
data[data['twp']=='LOWER MERION']['reason'].value_counts().head(10)
dfsc = data['station_code'].value_counts().head(10)

dfsc
data[data['station_code'] == "308A"]['reason_category'].value_counts()
data[data['station_code'] == "308A"]['reason'].value_counts().head(10)
plt.figure(figsize=(12,6))

plt.bar(dfsc.index,dfsc.values,width=0.6)

plt.title("Most Called Stations")

plt.xlabel("Station")

plt.ylabel("Number of calls")

plt.tight_layout()
dfzip = data['zip'].value_counts().head(10)

dfzip
data['zip'].nunique()
data[data['zip']==19401.0]['twp'].head(10)
data[data['zip']==19401.0]['reason_category'].value_counts()
data[data['zip']==19401.0]['reason'].value_counts().head()
data[data['zip']==19401.0].shape[0]
data['Date'].value_counts().head(10)
data["Year"].value_counts().head(1)
data['reason_category'].value_counts().head(5)
plt.figure(figsize=(12,6))

sns.countplot(x=data['reason_category'],data=data, palette='bright')

plt.title("Emergency call category")
plt.figure(figsize=(12,6))

sns.countplot(x=data['DayOfWeek'],data=data,hue=data['reason_category'],palette='bright')

plt.title("Emergency calls day wise groupby category")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
plt.figure(figsize=(12,6))

sns.countplot(x=data['reason_category'],data=data,hue=data['Year'],palette='bright')

plt.title("Emergency call category groupby year")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
plt.figure(figsize=(12,6))

sns.countplot(x=data['Year'],data=data,hue=data['reason_category'],palette='bright')

plt.title("Emergency calls yearly groupby category")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
plt.figure(figsize=(12,6))

sns.countplot(x=data['Month'],data=data,hue=data['reason_category'],palette='bright')

plt.title("Emergency call month wise groupby category")

plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
plt.figure(figsize=(12,6))

sns.countplot(x=data['Hour'],data=data,palette='Set2')

plt.title("Emergency call hour wise groupby category")
plt.figure(figsize=(15,8))

sns.countplot(x=data['Hour'],data=data,hue=data['reason_category'],palette='winter')

plt.title("Emergency call hour wise groupby category")

#plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
dfRes = data['reason'].value_counts().head(10)

dfRes
data['reason'].nunique()
plt.figure(figsize=(12, 6))

x = list(dfRes.index)

y = list(dfRes.values)

x.reverse()

y.reverse()



plt.title("Most emergency reasons of calls")

plt.ylabel("Reason")

plt.xlabel("Number of calls")



plt.barh(x,y)

plt.tight_layout()

plt.show()
byMonth = data.groupby('Month').count().sort_values(by='Month',ascending=True)

byMonth.head(12)
byMonth['twp'].plot(figsize=(12, 6))

plt.title('Count of calls per month')
plt.figure(figsize=(12, 8))

sns.lmplot(x='Month',y='twp',data=byMonth.reset_index())
byDate = data.groupby('Date').count().sort_values(by='Date',ascending=True)

byDate.head()
byDate['twp'].plot(figsize=(12,6))

plt.xticks(rotation=45)

plt.tight_layout()
data[data['reason_category']=='Traffic'].groupby('Date').count()['twp'].plot(figsize=(12,6))

plt.title('Traffic')

plt.tight_layout()
data[data['reason_category']=='Fire'].groupby('Date').count()['twp'].plot(figsize=(12,6))

plt.title('Fire')

plt.tight_layout()
data[data['reason_category']=='EMS'].groupby('Date').count()['twp'].plot(figsize=(12,6))

plt.title('EMS')

plt.tight_layout()
dayHour = data.groupby(['DayOfWeek','Hour']).count()['reason_category'].unstack()

dayHour
plt.figure(figsize=(12,6))

sns.heatmap(dayHour,cmap='viridis',linewidths=.1)
plt.figure(figsize=(12,6))

sns.clustermap(dayHour,cmap='viridis',linewidths=.1)
dayMonth = data.groupby(by=['DayOfWeek','Month']).count()['reason_category'].unstack()

dayMonth
plt.figure(figsize=(12,6))

sns.heatmap(dayMonth,cmap='viridis',linewidths=.1)
plt.figure(figsize=(12,6))

sns.clustermap(dayMonth,cmap='viridis',linewidths=.1)