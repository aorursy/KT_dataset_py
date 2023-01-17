# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
#Load the datasets



df_apr14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-apr14.csv")

df_may14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-may14.csv")

df_jun14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jun14.csv")

df_jul14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-jul14.csv")

df_aug14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-aug14.csv")

df_sep14=pd.read_csv("/kaggle/input/uber-pickups-in-new-york-city/uber-raw-data-sep14.csv")



#Merge the dataframes into one



df = df_apr14.append([df_may14,df_jun14,df_jul14,df_aug14,df_sep14], ignore_index=True)
df.head()
df.info()
#Renaming the Date/Time Colomn

df = df.rename(columns={'Date/Time': 'Date_time'})



#Converting the Date_time type into Datetime

df['Date_time'] = pd.to_datetime(df['Date_time'])



#Adding usufull colomns

df['Month'] = df['Date_time'].dt.month_name()

df['Weekday'] = df['Date_time'].dt.day_name()

df['Day'] = df['Date_time'].dt.day

df['Hour'] = df['Date_time'].dt.hour

df['Minute'] = df['Date_time'].dt.minute
df.head()
df.info()
df.describe(include = 'all')
#Grouping by Hour 

df_hour_grouped = df.groupby(['Hour']).count()



#Creating the sub dataframe

df_hour = pd.DataFrame({'Number_of_trips':df_hour_grouped.values[:,0]}, index = df_hour_grouped.index) 



df_hour.head()
df_hour.plot(kind='bar', figsize=(8,6))



plt.ylabel('Number of Trips')

plt.title('Trips by Hour')



plt.show()
#The highest number of trips by hour

max_Number_of_trips_hour = max(df_hour['Number_of_trips'])

max_hour = df_hour[df_hour['Number_of_trips'] == 336190].index[0]



print('The highest number of trips by hour is {} trip, that corresponds to the peak hour {}:00.'.format(max_Number_of_trips_hour, max_hour))

#Grouping by Month 

df_month_grouped = df.groupby(['Month'], sort=False).count()



#Creating the sub dataframe

df_month = pd.DataFrame({'Number_of_trips':df_month_grouped.values[:,0]}, index = df_month_grouped.index) 



df_month
df_month.plot(kind='bar', figsize=(8,6))



plt.ylabel('Number of Trips')

plt.title('Trips by Month')



plt.show()
number_of_trips_aug = df_month.loc['August'].values

number_of_trips_sep = df_month.loc['September'].values



ratio_month = (((number_of_trips_sep - number_of_trips_aug) / number_of_trips_aug) * 100)[0]

ratio_month = round(ratio_month)



print('The ratio of the increase from August to September is {} %.'.format(ratio_month))

#Grouping by Weekday

df_weekday_grouped = df.groupby(['Weekday'], sort = False).count()



#Creating the grouped DataFrame

df_weekday = pd.DataFrame({'Number_of_trips':df_weekday_grouped.values[:,0]}, index = df_weekday_grouped.index) 



df_weekday
df_weekday.plot(kind='bar', figsize=(8,6))



plt.ylabel('Number of Trips')

plt.title('Trips by Weekday')



plt.show()
#Getting the minimum number of trips by weekday

min_number_of_trips_weekday = min(df_weekday['Number_of_trips'])



#Getting the weekday where the number of trips is minimal

min_weekday = df_weekday[df_weekday['Number_of_trips'] == min_number_of_trips_weekday].index[0]



print('The lowest number of trips by weekday is {} trip, that corresponds to {}.'.format(min_number_of_trips_weekday, min_weekday))

#Getting the mean number of trips in the weekend - Non working day

mean_number_of_trips_weekend = ((df_weekday.loc['Saturday'] + df_weekday.loc['Sunday']) / 2).values



#Getting the mean number of trips for the rest of the week- Working day

mean_number_of_trips_workday = (((df_weekday.loc['Monday'] + df_weekday.loc['Tuesday'] + df_weekday.loc['Wednesday'] + df_weekday.loc['Thursday'] + df_weekday.loc['Friday'])/ 5).values)[0]



ratio_weekday = (((mean_number_of_trips_workday - mean_number_of_trips_weekend) / mean_number_of_trips_weekend) * 100)[0]

ratio_weekday = round(ratio_weekday, 1)



print('The mean number of trips during working days is {}% higher than the mean number of trips during weekends.'.format(ratio_weekday))
#Grouping by Day

df_day_grouped = df.groupby(['Day']).count()



#Creating the grouped DataFrame

df_day = pd.DataFrame({'Number_of_trips':df_day_grouped.values[:,0]}, index = df_day_grouped.index) 



df_day.head()
df_day.plot(kind='bar', figsize=(10,8))



plt.ylabel('Number of Trips')

plt.title('Trips by Day')



plt.show()
#Grouping by Hour and Month

df_hour_month_grouped = df.groupby(['Hour','Month']).count()



#Creating the grouped DataFrame

df_hour_month = pd.DataFrame({'Number_of_trips':df_hour_month_grouped.values[:,1]}, index = df_hour_month_grouped.index) 



df_hour_month.head(10)
#Reseting the Index

df_hour_month.reset_index(inplace= True)

df_hour_month.head()
#Preparing the Number of trips data

#We create a Numpy array that includes the Number of trips data then reshape it to fit our 

data_hour_month = df_hour_month['Number_of_trips'].values.reshape(24,6)

data_hour_month
df_hour_month = pd.DataFrame(data = data_hour_month, index = df_hour_month['Hour'].unique(), columns = df['Month'].unique())

df_hour_month.head()
df_hour_month.plot(kind='bar', figsize=(8,6), stacked=True)



plt.xlabel('Hour')

plt.ylabel('Number of Trips')

plt.title('Trips by Hour and Month')



plt.show()
df_hour_month.plot(kind='bar', figsize=(25,6),width=0.8)



plt.xlabel('Hour')

plt.ylabel('Number of Trips')

plt.title('Trips by Hour and Month')



plt.show()
#Grouping by Hour and weekday

df_weekday_hour_grouped = df.groupby(['Weekday','Hour'], sort = False).count()



#Creating the grouped DataFrame

df_weekday_hour = pd.DataFrame({'Number_of_trips':df_weekday_hour_grouped.values[:,1]}, index = df_weekday_hour_grouped.index) 



df_weekday_hour
#Reseting the Index

df_weekday_hour.reset_index(inplace= True)



#Preparing the Number of trips data

data_weekday_hour = df_weekday_hour['Number_of_trips'].values.reshape(7,24)



df_weekday_hour = pd.DataFrame(data = data_weekday_hour, index = df_weekday_hour['Weekday'].unique(), columns = df['Hour'].unique())

df_weekday_hour.head()
df_weekday_hour.plot(kind='bar', figsize=(20,6), width = 0.7)



plt.xlabel('Weekday')

plt.ylabel('Number of Trips')

plt.title('Trips by Hour and Weekday')



plt.show()
#Grouping by Weekday and Month

df_month_weekday_grouped = df.groupby(['Month','Weekday'], sort=False).count()



#Creating the grouped DataFrame

df_month_weekday = pd.DataFrame({'Number_of_trips':df_month_weekday_grouped.values[:,1]}, index = df_month_weekday_grouped.index) 



df_month_weekday.head(10)
#Reseting the Index

df_month_weekday.reset_index(inplace= True)



#Preparing the Number of trips 

data_month_weekday = df_month_weekday['Number_of_trips'].values.reshape(6,7)



df_month_weekday = pd.DataFrame(data = data_month_weekday, index = df_month_weekday['Month'].unique(), columns = df['Weekday'].unique())

df_month_weekday.head()
df_month_weekday.plot(kind='bar', figsize=(8,6), stacked = True)



plt.xlabel('Month')

plt.ylabel('Number of Trips')

plt.title('Trips by Month and Weekday')



plt.show()
df_month_weekday.plot(kind='bar', figsize=(18,6), width = 0.6)



plt.xlabel('Month')

plt.ylabel('Number of Trips')

plt.title('Trips by Month and Weekday')



plt.show()