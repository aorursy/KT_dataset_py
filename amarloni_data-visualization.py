# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

import numpy as np

import matplotlib

import seaborn as sns

%matplotlib inline

from sklearn.preprocessing import LabelEncoder

#le= LabelEncoder()

data=pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')

#data = le.fit_transform(data)

data.head()
data.info()
data.isnull().any().sum()
data.isna().sum()
df = pd.DataFrame(data)

df.drop(['End_Lat', 'End_Lng', 'Number', 'Wind_Chill(F)','Precipitation(in)'], axis=1, inplace=True)

df.dropna(axis=0)

df.head()
df.isnull().any().sum()
df.isnull().sum()
df =df.dropna(axis=0)


df.isnull().any().sum()

df.tail()

  

import matplotlib.pyplot as plt 

# plot a histogram  

data['Severity'].hist(bins=10) 

  

plt.figure(figsize =(20,5))

df.groupby(['State' ])['Severity'].sum().sort_values(ascending=False).head(10).plot.bar()
plt.figure(figsize =(10,5))

df.groupby(['Sunrise_Sunset'])['Severity'].size().sort_values(ascending=False).plot.pie()
plt.figure(figsize =(10,5))

df.groupby(['Stop', 'Station', 'Railway' ])['Severity'].size().sort_values(ascending=False).plot.bar()
plt.figure(figsize =(10,5))

df.groupby(['Give_Way', 'Crossing','Bump' ])['Severity'].sum().sort_values(ascending=False).plot.bar()


plt.figure(figsize =(20,5))

df.groupby(['Weather_Condition']).size().sort_values(ascending=False).head(12).plot.bar()
plt.figure(figsize =(12,5))

df.groupby(['Visibility(mi)']).size().sort_values(ascending=False).head(12).plot.bar()
import datetime

df['Start_Time']= pd.to_datetime(df['Start_Time'])

df['hour']= df['Start_Time'].dt.hour

df['year']= df['Start_Time'].dt.year

df['month']= df['Start_Time'].dt.month

df['week']= df['Start_Time'].dt.week

df['day']= df['Start_Time'].dt.weekday_name

df['quarter']= df['Start_Time'].dt.quarter

df['time_zone']= df['Start_Time'].dt.tz

df['time']= df['Start_Time'].dt.time
df.head()
plt.figure(figsize =(10,5))

df.groupby(['year']).size().sort_values(ascending=True).plot.bar()
plt.figure(figsize =(15,5))

df.groupby(['month']).size().plot.bar()
plt.figure(figsize =(15,5))

df.groupby(['year', 'month']).size().plot.bar()

plt.title('Number of accidents/year')

plt.ylabel('number of accidents')
df.groupby(['day']).size().plot.pie(figsize=(10,10))
plt.figure(figsize =(10,5))

df.groupby(['hour']).size().plot.bar()

plt.title('At which hour of day accidents happen')

plt.ylabel('count of accidents')
df['day_zone'] = pd.cut((df['hour']),bins=(0,6,12,18,24), labels=["night", "morning", "afternoon", "evening"])

plt.figure(figsize =(10,5))

df.groupby(['day_zone']).size().plot.bar()