# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import plotly


import seaborn as sns

# Any results you write to the current directory are saved as output.
emergency_calls = pd.read_csv('../input/911.csv')
emergency_calls.head()
emergency_calls['title_type'] = emergency_calls.title.str.split(':').str[0]

emergency_calls.head()

emergency_calls.groupby('title_type')['title_type'].count()
#pd.crosstab(emergency_calls.title_type,emergency_calls.title).apply(lambda r: r.count(), axis=1)

labels=['EMS','Fire','Traffic']

colors = ['#ff9999','#66b3ff','#99ff99','#ffcc99']


fig1, ax1 = plt.subplots()

ax1.pie(emergency_calls.groupby('title_type')['title_type'].count(),labels=labels,colors = colors, autopct='%1.1f%%', pctdistance=1.1, labeldistance=1.2)
ax1.axis('equal')

plt.tight_layout()
plt.show()
top_10_towns = emergency_calls.groupby('twp')['zip'].count().nlargest(10)


top_10_towns.plot.bar()

top_10_towns = pd.DataFrame(top_10_towns)


top_10_towns = top_10_towns.reset_index()

top_10_titles = emergency_calls.groupby(['twp','title_type'])['zip'].count()

top_10_titles = pd.DataFrame(top_10_titles)

top_10_titles = top_10_titles.reset_index()

list(top_10_titles)


#top_10_titles.plot.bar().
top_5 = pd.merge(top_10_towns,top_10_towns,how='left',on='twp')


top_5 = top_5.iloc[:5,0]


top_5_data = top_10_titles[top_10_titles.twp.isin(top_5)]

top_5_data

ax = sns.factorplot(x="twp", y="zip",hue='title_type', data=top_5_data,kind="bar",size=4, aspect=2)

ax.set_titles('Distribution of Incidents across the top 5 Towns')
ax.set(xlabel='Towns', ylabel='Count of Incidents')



emergency_calls['month'] = emergency_calls.timeStamp.str.split('-').str[1]

emergency_calls['month'] = emergency_calls.month.str.split('-').str[0]

emergency_calls.head()

month_count = emergency_calls.groupby('month')['zip'].count()

month_count = pd.DataFrame(month_count)

month_count = month_count.reset_index()

list(month_count)

ax = sns.factorplot(x="month", y="zip", data=month_count,kind="bar",size=4, aspect=2)

ax.set_titles("Month wise Incident Plot")

ax.set_xlabels("Month")

ax.set_ylabels("Count of Incidents")

emergency_calls['time'] = emergency_calls.timeStamp.str.split('-').str[2]

emergency_calls['time'] = emergency_calls.time.str.split(' ').str[1]

emergency_calls['time'] = emergency_calls.time.str.split(':').str[0]

emergency_calls['time'] .head()

time_count = emergency_calls.groupby('time')['zip'].count()

time_count = pd.DataFrame(time_count)

time_count = time_count.reset_index()

ax = sns.factorplot(x="time", y="zip", data=time_count,kind="bar",size=4, aspect=2)

ax.set_titles("Month wise Incident Plot")

ax.set_xlabels("Time")

ax.set_ylabels("Count of Incidents")
