# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sb

import glob

import os

import datetime

import math



%matplotlib inline



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df =pd.read_csv('../input/ford-go-bike-dataset-for-year-2018/combined_data.csv')

df.head()
df.shape
df.info()
df.duplicated().sum() #finding out duplicate rows if any
df.isnull().sum() #finding out null values for each column
df.describe()
#create copy of original dataframe

df_clean = df.copy()
df_clean.start_time = pd.to_datetime(df_clean.start_time) # changing start time column type to datetime

df_clean.end_time = pd.to_datetime(df_clean.end_time) #changing end time column type to datetime

df_clean.drop('bike_share_for_all_trip',axis=1,inplace=True) #dropping the column

df_clean.drop('start_station_id',axis=1,inplace=True) # dropping the column

df_clean.drop('end_station_id',axis=1,inplace=True) #dropping the column

df_clean.info()
# extract month number from start_time column

df_clean['month']=df_clean['start_time'].dt.month.astype(int)

# extract weekdays from start_time column

df_clean['weekday']=df_clean['start_time'].dt.strftime('%a')

# extract day from start_time column

df_clean['day']=df_clean['start_time'].dt.day.astype(int)

# extract hour from start_time column

df_clean['hour']=df_clean['start_time'].dt.hour

#adding trip duration in minutes coulmn

df_clean['duration_min']=df_clean['duration_sec']/60

#adding distance column

df_clean['distance'] = np.sqrt((df_clean.start_station_longitude - df_clean.end_station_longitude) ** 2 + (df_clean.start_station_latitude - df_clean.end_station_latitude) ** 2)
df_clean.info()
df_clean.head(5)
plt.figure(figsize=[8, 16])



plt.subplot(3,1,1)

count2=df_clean.groupby('month').size()

sb.pointplot(data = df_clean,x = count2.index,y=count2,color=sb.color_palette()[0]);

plt.title("Distribution of bike rides according to month");

plt.ylabel("No.of bike trips");

plt.xlabel("Month");



#according to day

plt.subplot(3,1,2)

count1=df_clean.groupby('weekday').size()

weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

sb.pointplot(data=df_clean, x=count1.index, y=count1 ,color = sb.color_palette()[0], order = weekday);

plt.title("Distribution of bike rides according to day");

plt.ylabel("No.of bike trips");

plt.xlabel("Day");



#according to hour

count=df_clean.groupby('hour').size()

plt.subplot(3,1,3)

sb.pointplot(data = df_clean,x =count.index ,y=count, color=sb.color_palette()[0]);

plt.title("Distribution of hourly bike rides");

plt.ylabel("No.of bike trips");

plt.xlabel("Hour of the day");
# Here we plot the distribution of trip durations.

binedges = np.arange(0, df_clean['duration_min'].max() + 100 , 100)

plt.hist(data = df_clean , x = 'duration_sec' , bins = binedges)

plt.xlim(0,3000);

plt.title('Distribution of various Trip Duration(sec)');

plt.xlabel('Duration in seconds');

plt.ylabel('Frequency');
# plotting the log values of trip duration

binedges = 10**np.arange(0 , 3+0.1 , 0.1)

plt.hist(data = df_clean , x = 'duration_min' , bins = binedges);

plt.title('Distribution of various Trip Duration')

plt.xlabel('Duration in Minutes');

plt.ylabel('Frequency')

plt.xscale('log');
customer = df_clean.query('user_type == "Customer"')['bike_id'].count()

subscriber = df_clean.query('user_type == "Subscriber"')['bike_id'].count()





customer_prop = customer / df_clean['bike_id'].count()

subscriber_prop = subscriber / df_clean['bike_id'].count()





labels = ['Customer', 'Subscriber']

sizes = [customer_prop, subscriber_prop]

print("Customer proportion: "+str(customer_prop))

print("Subscriber proportion: "+str(subscriber_prop))



sb.barplot(data=df_clean,x=labels,y=sizes);

plt.ylabel("Proportion of bike rides");

plt.xlabel("User Type");

plt.title("Bikes Ride Proportion by User Type");

print(df_clean.groupby('user_type')['duration_sec'].mean())

plt.figure(figsize = [10, 5]);

base_color = sb.color_palette()[1]

sb.boxplot(data = df_clean, x = 'user_type', y = 'duration_sec', color = base_color);

plt.title('Trip Duration and User Type');

plt.ylim([-10, 3000]);

plt.xlabel('User Type');

plt.ylabel('Duration_sec');

plt.show();
plt.figure(figsize = [12,10])

plt.scatter(data = df_clean , x = 'month' , y = 'duration_min' , alpha = 0.2);

plt.title('Anaylsing Trip Duration in minutes Trend According to Month');

plt.xlabel('Months');

plt.ylabel('Trip Duration in minutes');
plt.figure(figsize=[10, 4]);

# plot the point plot of day vs user type

weekday = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

sb.pointplot(data=df_clean, x='weekday', y='duration_min' ,color = sb.color_palette()[0], hue='user_type',order = weekday);

plt.ylabel("Average trip duration in minutes");

plt.xlabel("Day of the week");

plt.legend(title='User Type');

plt.title("Average duration of trip per day for each user type");

print(df_clean.groupby(['weekday','user_type'])['duration_min'].mean())
plt.figure(figsize=[10, 4]);



# plot the point plot of month vs user type

sb.pointplot(data=df_clean, x='month', y='duration_min', hue='user_type',color = sb.color_palette()[0]);

plt.xlabel('Months of the year');

plt.ylabel('Average trip Duration in minutes');

plt.legend(title='User Type');

plt.title("Average duration of trips in each month for each user type");

print(df_clean.groupby(['month','user_type'])['duration_min'].mean())
g = sb.FacetGrid(data =  df_clean, hue = 'user_type', height = 8)

g.map(plt.scatter, 'month','distance', alpha = 0.3)

g.add_legend(title="User Type");

plt.title('Trip Distance Against Month and User Type');

plt.xlabel('Month');

plt.ylabel('Trip Distance');

plt.ylim(-0.01,0.2);