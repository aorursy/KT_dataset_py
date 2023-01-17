import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(20,20))
trainsample=pd.read_csv('../input/train_sample.csv')
trainsample.shape
trainsample.dtypes
trainsample.is_attributed=trainsample.is_attributed.astype('object')
trainsample.app=trainsample.app.astype('object')
trainsample.os=trainsample.os.astype('object')
trainsample.device=trainsample.device.astype('object')
trainsample.channel=trainsample.channel.astype('object')

trainsample.click_time=pd.to_datetime(trainsample['click_time'],format='%Y-%m-%d %H:%M:%S')
trainsample.attributed_time=pd.to_datetime(trainsample['attributed_time'],format='%Y-%m-%d %H:%M:%S')


trainsample.isnull().sum()
trainsample.is_attributed.value_counts()
def GroupByColumns(columns):
    groupCol = trainsample.groupby(columns)\
                    .size()\
                    .sort_values(ascending=False)\
                    .reset_index()
    return groupCol
columns=['app','device','os']
GroupByColumns(columns)
columns=['app','device','os','is_attributed']
GroupByColumns(columns)
columns=['app','device','os','is_attributed']
groupCol = trainsample.groupby(columns)\
                    .size()\
                    .sort_values(ascending=False)\
                    .reset_index()
groupCol
AttributedDF= trainsample[trainsample.is_attributed==1]
timeDiffList= list()
i= 0;
def timeBtwDwnld(df):
    for i in range(0, len(df)):
        timeDiffList.append((df.iloc[i]['attributed_time']-df.iloc[i]['click_time']).seconds)
    return timeDiffList
    
AttributedDF['timediff']=timeBtwDwnld(AttributedDF)
plt.hist(AttributedDF['timediff'],bins=30)
plt.show()
AttributedDF[AttributedDF.timediff>3600].shape
appseries=trainsample.app.value_counts().nlargest(20)
plt.figure(figsize=(20,20))
appseries.plot.bar()
plt.show()
osseries=trainsample.os.value_counts().nlargest(20)
plt.figure(figsize=(20,20))
osseries.plot.bar()
plt.show()
deviceseries=trainsample.device.value_counts().nlargest(20)
plt.figure(figsize=(20,20))
deviceseries.plot.bar()
plt.show()
channelseries=trainsample.channel.value_counts().nlargest(20)
plt.figure(figsize=(20,20))
channelseries.plot.bar()
plt.show()
times = pd.DatetimeIndex(trainsample.click_time)
plt.figure(figsize=(20,20))
grouped = trainsample.groupby([times.day])['ip'].count()
grouped.plot.bar()
plt.show()
grouped = trainsample.groupby([times.hour])['ip'].count()
plt.figure(figsize=(20,20))
grouped.plot.bar()
plt.show()
times = pd.DatetimeIndex(trainsample.attributed_time)
grouped = trainsample.groupby([times.day])['ip'].count()
plt.figure(figsize=(20,20))
grouped.plot.bar()
plt.show()
grouped = trainsample.groupby([times.hour])['ip'].count()
plt.figure(figsize=(20,20))
grouped.plot.bar()
plt.show()
grouped = trainsample.groupby([times.minute])['ip'].count()
plt.figure(figsize=(20,20))
grouped.plot.bar()
plt.show()
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler()

orig_x_data=trainsample[['ip','os','device','app','channel']]
orig_y_data=trainsample[['is_attributed']]

x_oversampled, y_oversampled = ros.fit_sample(orig_x_data, orig_y_data)
from collections import Counter
print(sorted(Counter(y_oversampled).items()))

trainsample['clicktimemins'] =trainsample.click_time.map(lambda t: t.strftime('%Y-%m-%d %H:%M'))
clickcountbymin=trainsample.groupby(['clicktimemins'])['ip'].count()
plt.figure(figsize=(15,15))
clickcountbymin.plot()
plt.show()
from pandas import Series
from pandas import DataFrame
from pandas import Grouper
import numpy as np
series=Series(clickcountbymin)
n = 480
newdf=trainsample.set_index(['click_time'])
dtresampler=newdf['ip'].resample('60s')
trainsample['clickdate'] =trainsample.click_time.map(lambda t: t.strftime('%Y-%m-%d'))
trainsample['timeofclick'] =trainsample.click_time.map(lambda t: t.strftime('%H:%M:%S'))
uniqdates=trainsample['clickdate'].unique()
df6th=trainsample[trainsample.clickdate=='2017-11-06']
df6th=df6th.set_index(['click_time'])
df7th=trainsample[trainsample.clickdate=='2017-11-07']
df7th=df7th.set_index(['click_time'])
df8th=trainsample[trainsample.clickdate=='2017-11-08']
df8th=df8th.set_index(['click_time'])
df9th=trainsample[trainsample.clickdate=='2017-11-09']
df9th=df9th.set_index(['click_time'])

fig,ax=plt.subplots(4,1,sharex='row',sharey=False,squeeze=False)
plt.figure(figsize=(20,20))
ax[0][0]

ax[0][0].plot(df6th['ip'].resample('60s').count())

ax[1][0].plot(df7th['ip'].resample('60s').count())

ax[2][0].plot(df8th['ip'].resample('60s').count())

ax[3][0].plot(df9th['ip'].resample('60s').count())
plt.show()
columns=['ip','app','device','os']
GroupByColumns(columns)
#extract hour as a feature
trainsample['click_hour']=trainsample['click_time'].dt.hour
trainsample.is_attributed=trainsample.is_attributed.astype('int')
#thanks to yuliagm for the conversation rate  idea 
proportion = trainsample[['ip', 'is_attributed']].groupby('ip', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = trainsample[['ip', 'is_attributed']].groupby('ip', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='ip', how='left')
merge.columns = ['ip', 'click_count', 'prop_downloaded']

ax = merge[:300].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates  300 Most Popular IPs')
ax.set(ylabel='Click Count')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular IPs')
print(merge[:30])
trainsample['click_hour']=trainsample['click_time'].dt.hour
sns.barplot('click_hour', 'is_attributed', data=trainsample)
plt.title('HOURLY CONVERSION RATIO');
plt.ylabel('Converted Ratio');
plt.show()
proportion = trainsample[['app', 'is_attributed']].groupby('app', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = trainsample[['app', 'is_attributed']].groupby('app', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='app', how='left')
merge.columns = ['app', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates over Counts of 100 Most Popular Apps')
ax.set(ylabel='Click Count')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular Apps')
print(merge[:20])
proportion = trainsample[['os', 'is_attributed']].groupby('os', as_index=False).mean().sort_values('is_attributed', ascending=False)
counts = trainsample[['os', 'is_attributed']].groupby('os', as_index=False).count().sort_values('is_attributed', ascending=False)
merge = counts.merge(proportion, on='os', how='left')
merge.columns = ['os', 'click_count', 'prop_downloaded']

ax = merge[:100].plot(secondary_y='prop_downloaded')
plt.title('Conversion Rates ')
ax.set(ylabel='Click Count')
plt.ylabel('Proportion Downloaded')
plt.show()

print('Counversion Rates over Counts of Most Popular os')
print(merge[:20])
