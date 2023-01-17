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
dff = pd.read_csv("../input/911.csv")

dff['Reasons'] = dff['title'].apply(lambda x:x.split(":")[0])
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
sns.set_style('whitegrid')

sns.countplot(x='Reasons',data=dff)
dff['timeStamp'] = pd.to_datetime(dff['timeStamp'])

dff['Hour'] = dff['timeStamp'].apply(lambda x:x.hour)

dff['Month'] = dff['timeStamp'].apply(lambda x:x.month)

dff['Day of Weak'] = dff['timeStamp'].apply(lambda x:x.weekday())

dff.head()
dmap = {0:'Mon',1:'Tue',2:'Wed',3:'Thu',4:'Fri',5:'Sat',6:'Sun'}

dff['Day of Weak'] = dff['Day of Weak'].map(dmap)
sns.countplot(x='Day of Weak',data=dff,hue='Reasons')

plt.legend(bbox_to_anchor=(1.1, 1.05))
sns.countplot(x='Month',data=dff,hue='Reasons')

plt.legend(bbox_to_anchor=(1.25, 1.05))
group_by_month = dff.groupby(['Month']).count()

group_by_month['Month'] = group_by_month.index

group_by_month
ax = sns.lineplot(x='Month',y = 'e',data=group_by_month)

ax.set_title('Calls to 911 month wise')
sns.lmplot(x='Month',y='e',data=group_by_month)

dff['date'] = dff['timeStamp'].apply(lambda x:x.date())

dff.head()
group_by_date = dff.groupby(['date']).count()

group_by_date['date'] = group_by_date.index

group_by_date.head()
plt.figure(figsize=(20,7))

ax = sns.lineplot(x='date',y='e',data=group_by_date)

ax.set_title('calls to 911 date wise')

df_traffic = dff[dff.Reasons == 'Traffic']

df_traffic.head()

df_traffic['date'] = df_traffic['timeStamp'].apply(lambda x:x.date())

df_traffic_by_date = df_traffic.groupby(['date']).count()

df_traffic_by_date['date'] = df_traffic_by_date.index

plt.figure(figsize=(20,7))

ax = sns.lineplot(x='date',y='e',data=df_traffic_by_date)

ax.set_title('calls to 911 due to Traffic date wise')
df_fire = dff[dff.Reasons == 'Fire']

df_fire['date'] = df_fire['timeStamp'].apply(lambda x:x.date())

df_fire_by_date = df_fire.groupby(['date']).count()

df_fire_by_date['date'] = df_fire_by_date.index

plt.figure(figsize=(20,7))

ax = sns.lineplot(x='date',y='e',data=df_fire_by_date)

ax.set_title('calls to 911 due to Fire date wise')
df_ems = dff[dff.Reasons == 'EMS']

df_ems['date'] = df_ems['timeStamp'].apply(lambda x:x.date())

df_ems_by_date = df_ems.groupby(['date']).count()

df_ems_by_date['date'] = df_ems_by_date.index

plt.figure(figsize=(20,7))

ax = sns.lineplot(x='date',y='e',data=df_ems_by_date)

ax.set_title('calls to 911 due to EMS date wise')
DayHour = dff.groupby(by=['Day of Weak','Hour']).count()['Reasons'].unstack()
plt.figure(figsize=(10,7))

sns.heatmap(DayHour)