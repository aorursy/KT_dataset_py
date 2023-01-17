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
import seaborn as sns
%matplotlib inline
data = pd.read_csv('/kaggle/input/whatsapp-chat/Whatsapp_chat.csv', index_col=0)
data # there are 51575 messages in a period from 19.05 to 12.08 (day/month)
data.groupby('names').count()
from matplotlib import ticker
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True) 
formatter.set_powerlimits((3,3))
sns.countplot(x='names', data=data).yaxis.set_major_formatter(formatter) #shows how many messages that every person sent 
data_temp = data['names'].value_counts()
data_temp
data_temp.plot.pie(figsize=(12,12), legend=False, autopct='%1.1f%%', shadow=True, explode=(0.05, 0, 0, 0)).set(ylabel='message ratios')
plt.figure(figsize=(10,5))
sns.countplot(x='hours', data=data).yaxis.set_major_formatter(formatter) #shows distribution among hours of a day
data_by_date = data.groupby(['months', 'days']).count()
data_by_date
data_heat = data_by_date.pivot_table(values='hours', index='months', columns='days')
plt.figure(figsize=(30,4))
sns.heatmap(data_heat, cmap='rainbow').set(title='Message Density of Each Day')
data_by_timestamp = data[['timestamp', 'names']].value_counts().unstack(level=1).fillna(0) #groups data frame by timestamp and shows how many messages are sent by each person
data_by_timestamp
plt.figure(figsize=(20,10))
'''ax = sns.lineplot(data=data_by_timestamp["Person1"])
ax = sns.lineplot(data=data_by_timestamp["Person2"])
ax = sns.lineplot(data=data_by_timestamp["Person3"])
ax = sns.lineplot(data=data_by_timestamp["Person4"])'''
ax = sns.lineplot(data=data_by_timestamp)
ax.xaxis.set_ticks(np.arange(0, 100, 10))
ax
#extracting column values for further examination
names = np.array(data['names'])
hours = np.array(data['hours'])
dates = np.array(data['timestamp'])
#according to names and dates, examines columns and counts how many consecutive messages are sent by only two people in an ongoing conversation
#and indicates in which time interval that messages are taken place
#the message number must be greater than 49 to be considered as a streak
date_list = []
name_pair = []
time_interval = []
count = 0
name_pair_list = []
streak_list = []
time_interval_list = []
index = 0
for name in names:
    if len(name_pair) == 0:
        count += 1
        name_pair.append(name)
        date = dates[index]
        time_interval.append(hours[index])
    elif len(name_pair) == 1:
        count += 1
        if name not in name_pair:
            name_pair.append(name)
            if len(time_interval) == 2:
                time_interval[1] = hours[index]
            else:
                time_interval.append(hours[index])
    elif name in name_pair:
        time_interval[1] = hours[index]
        count += 1
    else:
        if count>49:
            streak_list.append(count)
            copy_pair = sorted(name_pair)
            name_pair_list.append((copy_pair[0], copy_pair[1]))
            time_interval_list.append((time_interval[0], time_interval[1]))
            date_list.append(date)
        count = 1
        name_pair.pop(0)
        time_interval.pop(0)
        name_pair.append(name)
        time_interval.append(hours[index])
        date = dates[index]
    index += 1
#new DataFrame that shows how many messages are sent by people pairs as streaks
data_streaks = pd.DataFrame({'Name Pairs':name_pair_list, 'Streak Counts':streak_list, 'Time Interval':time_interval_list, 'Timestamp':date_list})
data_streaks.sort_values(by='Streak Counts', ascending=False)
#shows how many total messages are sent by streaks between each person 
plt.figure(figsize=(10,5))
axes = sns.barplot(x='Name Pairs', y ='Streak Counts', data=data_streaks, estimator=np.sum)
axes.set(ylabel='Sum of Streak Numbers')
axes.yaxis.set_major_formatter(formatter)