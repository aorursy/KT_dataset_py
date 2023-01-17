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
data = pd.read_csv('/kaggle/input/crowdedness-at-the-campus-gym/data.csv')

data.head()
data.isnull().sum()
from scipy import stats

sns.distplot(data['number_people'],fit=stats.norm)
data.dtypes
data[data['number_people'] == 0].shape
data[['date','time']] = data['date'].str.split(expand=True)
data.head()
data.dtypes
data[['time','temp']] = data.time.apply( 

   lambda x: pd.Series(str(x).split("-"))) 
data.head()
data.drop(['temp','timestamp'],axis=1,inplace=True)
import plotly.express as px
number_of_people_visiting_each_day = data.groupby('date').agg({'number_people':'sum'})
number_of_people_visiting_each_day.head()
fig = px.line(number_of_people_visiting_each_day,number_of_people_visiting_each_day.index,number_of_people_visiting_each_day['number_people'])

fig.show()
number_of_people_visiting_each_day['moving_avg'] = number_of_people_visiting_each_day['number_people'].rolling(window=30).mean()
fig = px.line(number_of_people_visiting_each_day,number_of_people_visiting_each_day.index,number_of_people_visiting_each_day['moving_avg'])

fig.show()
number_of_people_visiting_10_mins = data.groupby('time').agg({'number_people':'sum'})
fig = px.line(number_of_people_visiting_10_mins,number_of_people_visiting_10_mins.index,number_of_people_visiting_10_mins['number_people'])

fig.show()
number_of_people_visiting_days_week = data.groupby(['day_of_week']).agg({'number_people':'sum'})
number_of_people_visiting_days_week
fig = px.line(number_of_people_visiting_days_week,number_of_people_visiting_days_week.index,number_of_people_visiting_days_week['number_people'])

fig.show()
data.head()
is_weekend_vs_number_people = data.groupby('is_weekend')['number_people'].sum()
fig = px.pie(is_weekend_vs_number_people, values='number_people', names=is_weekend_vs_number_people.index)

fig.show()
is_holiday_vs_number_people = data.groupby('is_holiday')['number_people'].sum()
fig = px.pie(is_holiday_vs_number_people, values='number_people', names=is_holiday_vs_number_people.index)

fig.show()
fig = px.scatter(data, x="number_people", y="temperature")

fig.show()
sns.jointplot(data=data,x='number_people',y='temperature')
number_of_people_vs_hour = data.groupby('hour').agg({'number_people':'sum'})
fig = px.line(number_of_people_vs_hour,number_of_people_vs_hour.index,number_of_people_vs_hour['number_people'])

fig.show()
number_of_people_vs_month = data.groupby('month').agg({'number_people':'sum'})
fig = px.line(number_of_people_vs_month,number_of_people_vs_month.index,number_of_people_vs_month['number_people'])

fig.show()
data.head()
start_semester = data[['number_people','date','is_start_of_semester','time']]
start_semester_groupby = start_semester.groupby(['date','is_start_of_semester']).agg({'number_people':'sum'})
start_semester_0 = start_semester[start_semester['is_start_of_semester'] == 0]
start_semester_0_groupby = start_semester_0.groupby('date').agg({'number_people':'sum'})
fig = px.line(start_semester_0_groupby,start_semester_0_groupby.index,start_semester_0_groupby['number_people'])

fig.show()
start_semester_1 = start_semester[start_semester['is_start_of_semester'] == 1]

start_semester_1_groupby = start_semester_1.groupby('date').agg({'number_people':'sum'})
fig = px.line(start_semester_1_groupby,start_semester_1_groupby.index,start_semester_1_groupby['number_people'])

fig.show()


del start_semester_0

del start_semester_0_groupby

del start_semester_1

del start_semester_1_groupby
import gc

gc.collect()
start_semester_0 = start_semester[start_semester['is_start_of_semester'] == 0]

start_semester_1 = start_semester[start_semester['is_start_of_semester'] == 1]

del start_semester_0

del start_semester_1
start_semester_vs_number_people = start_semester.groupby('is_start_of_semester')['number_people'].sum()
fig = px.pie(start_semester_vs_number_people, values='number_people', names=start_semester_vs_number_people.index)

fig.show()
fig = px.line(start_semester, x="date", y="number_people", color='is_start_of_semester')

fig.show()
start_semester['moving_avg'] = start_semester['number_people'].rolling(window=50).mean()
start_semester.head(50)
fig = px.line(x=start_semester['date'][49:],y = start_semester['moving_avg'][49:])

fig.show()
start_semester['moving_avg_100'] = start_semester['number_people'].rolling(window=100).mean()
fig = px.line(x=start_semester['date'][99:],y = start_semester['moving_avg_100'][99:],color=start_semester['is_start_of_semester'][99:])

fig.show()
data.head()
during_semester_vs_number_people = data.groupby('is_during_semester')['number_people'].sum()
fig = px.pie(during_semester_vs_number_people, values='number_people', names=during_semester_vs_number_people.index)

fig.show()
data.head()
data_corr = data.corr()
sns.heatmap(data_corr)
data_corr['number_people']
data.drop('is_holiday',axis=1,inplace=True)
sns.distplot(data['number_people'])
import statsmodels.api as sm 

sm.qqplot(data['number_people'], line ='45') 

plt.show()
data['number_people_log'] = np.log1p(data['number_people'])
sns.distplot(data['number_people_log'])