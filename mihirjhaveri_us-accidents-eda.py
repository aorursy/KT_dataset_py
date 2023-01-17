import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import matplotlib.pyplot as plt

import seaborn as sns
us_accidents_dec19_df = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')

us_accidents_dec19_df.head()
us_accidents_dec19_df.dtypes
us_accidents_dec19_df.shape
us_accidents_dec19_df.describe()
#We can make the start and End Times real datetime columns

us_accidents_dec19_df['Start_Time'] = pd.to_datetime(us_accidents_dec19_df['Start_Time'])

us_accidents_dec19_df['End_Time'] = pd.to_datetime(us_accidents_dec19_df['End_Time'])

us_accidents_dec19_df.dtypes
#Lets identify columns with Nulls and NaNs.



us_accidents_dec19_df.isnull().sum()
#Cleanup some columns by removing Nulls or Filling NaNs

#Define a funcation impute median

def impute_median(series):

    return series.fillna(series.median())



def impute_mean(series):

    return series.fillna(series.mean())
us_accidents_dec19_df.End_Lat = us_accidents_dec19_df['End_Lat'].transform(impute_median)

us_accidents_dec19_df.End_Lng = us_accidents_dec19_df['End_Lng'].transform(impute_median)

us_accidents_dec19_df.TMC = us_accidents_dec19_df['TMC'].transform(impute_median)

us_accidents_dec19_df.Number = us_accidents_dec19_df['Number'].transform(impute_mean)

us_accidents_dec19_df['Temperature(F)'] = us_accidents_dec19_df['Temperature(F)'].transform(impute_median)

us_accidents_dec19_df['Wind_Chill(F)'] = us_accidents_dec19_df['Wind_Chill(F)'].transform(impute_median)

us_accidents_dec19_df['Humidity(%)'] = us_accidents_dec19_df['Humidity(%)'].transform(impute_median)

us_accidents_dec19_df['Pressure(in)'] = us_accidents_dec19_df['Pressure(in)'].transform(impute_median)

us_accidents_dec19_df['Visibility(mi)'] = us_accidents_dec19_df['Visibility(mi)'].transform(impute_median)

us_accidents_dec19_df['Wind_Speed(mph)'] = us_accidents_dec19_df['Wind_Speed(mph)'].transform(impute_median)

us_accidents_dec19_df['Precipitation(in)'] = us_accidents_dec19_df['Precipitation(in)'].transform(impute_median)

us_accidents_dec19_df.isnull().sum()
#Modes of categorical values

print(us_accidents_dec19_df['Weather_Condition'].mode())

print(us_accidents_dec19_df['Astronomical_Twilight'].mode())

print(us_accidents_dec19_df['Nautical_Twilight'].mode())

print(us_accidents_dec19_df['Weather_Timestamp'].mode())

print(us_accidents_dec19_df['Civil_Twilight'].mode())

print(us_accidents_dec19_df['Sunrise_Sunset'].mode())

print(us_accidents_dec19_df['Wind_Direction'].mode())

print(us_accidents_dec19_df['City'].mode())

print(us_accidents_dec19_df['Zipcode'].mode())

print(us_accidents_dec19_df['Airport_Code'].mode())

print(us_accidents_dec19_df['Timezone'].mode())

print(us_accidents_dec19_df['Description'].mode())
us_accidents_dec19_df['Weather_Condition'].fillna(str(us_accidents_dec19_df['Weather_Condition'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Astronomical_Twilight'].fillna(str(us_accidents_dec19_df['Astronomical_Twilight'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Nautical_Twilight'].fillna(str(us_accidents_dec19_df['Nautical_Twilight'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Weather_Timestamp'].fillna(str(us_accidents_dec19_df['Weather_Timestamp'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Weather_Timestamp'] = pd.to_datetime(us_accidents_dec19_df['Weather_Timestamp'])

us_accidents_dec19_df['Civil_Twilight'].fillna(str(us_accidents_dec19_df['Civil_Twilight'].mode().values[0]), inplace=True)

us_accidents_dec19_df['City'].fillna(str(us_accidents_dec19_df['City'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Sunrise_Sunset'].fillna(str(us_accidents_dec19_df['Sunrise_Sunset'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Wind_Direction'].fillna(str(us_accidents_dec19_df['Wind_Direction'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Zipcode'].fillna(str(us_accidents_dec19_df['Zipcode'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Airport_Code'].fillna(str(us_accidents_dec19_df['Airport_Code'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Timezone'].fillna(str(us_accidents_dec19_df['Timezone'].mode().values[0]), inplace=True)

us_accidents_dec19_df['Description'].fillna(str(us_accidents_dec19_df['Description'].mode().values[0]), inplace=True)

us_accidents_dec19_df.isnull().sum()
us_accidents_state_wise_counts = us_accidents_dec19_df.groupby('State')['ID'].count().reset_index()

us_accidents_state_wise_counts
us_accidents_state_wise_counts.shape
us_accidents_state_wise_counts = us_accidents_state_wise_counts.sort_values(by = "ID",ascending=False)

us_accidents_state_wise_counts
sns.set(style="whitegrid")

f, ax = plt.subplots(figsize=(6, 15))

sns.barplot(y="State", x="ID", data=us_accidents_state_wise_counts)
f, ax = plt.subplots(figsize=(18, 8))

ax = sns.lineplot(x="State", y="ID", data=us_accidents_state_wise_counts)

ax.set(ylabel='Total no of accidents')

plt.show()



plt.figure(figsize =(20,5))

us_accidents_dec19_df.groupby(['State' ])['Severity'].sum().sort_values(ascending=False).head(10).plot.bar()
us_accidents_dec19_df['hour']= us_accidents_dec19_df['Start_Time'].dt.hour

us_accidents_dec19_df['year']= us_accidents_dec19_df['Start_Time'].dt.year

us_accidents_dec19_df['month']= us_accidents_dec19_df['Start_Time'].dt.month

us_accidents_dec19_df['week']= us_accidents_dec19_df['Start_Time'].dt.week

us_accidents_dec19_df['day']= us_accidents_dec19_df['Start_Time'].dt.day_name()

us_accidents_dec19_df['quarter']= us_accidents_dec19_df['Start_Time'].dt.quarter

us_accidents_dec19_df['time_zone']= us_accidents_dec19_df['Start_Time'].dt.tz

us_accidents_dec19_df['time']= us_accidents_dec19_df['Start_Time'].dt.time
plt.figure(figsize =(15,5))

us_accidents_dec19_df.groupby(['year', 'month']).size().plot.bar()

plt.title('Number of accidents/year')

plt.ylabel('number of accidents')
us_accidents_dec19_df.groupby(['day']).size().plot.pie(figsize=(10,10))
plt.figure(figsize =(10,5))

us_accidents_dec19_df.groupby(['hour']).size().plot.bar()

plt.title('At which hour of day accidents happen')

plt.ylabel('count of accidents')
us_accidents_dec19_df['day_zone'] = pd.cut((us_accidents_dec19_df['hour']),bins=(0,6,12,18,24), labels=["night", "morning", "afternoon", "evening"])

plt.figure(figsize =(10,5))

us_accidents_dec19_df.groupby(['day_zone']).size().plot.bar()
weather_condition = us_accidents_dec19_df.groupby('Weather_Condition').count()

weather_condition
accident_state_wise = us_accidents_dec19_df.groupby('State').count()

accident_state_wise
print('The State where accidents usually occur the Most in the US? : ',accident_state_wise.Number.idxmax())

print('The Rank of Accidents state wise is:\n ',accident_state_wise['Number'].sort_values(ascending=False))
#plotting the Lat against Long could show the map of the area

plt.figure(figsize=(50,30))

plt.title('Most Hits per Area')

plt.xlabel('Start Longitude')

plt.ylabel('Start Latitude')

plt.plot(us_accidents_dec19_df.Start_Lng, us_accidents_dec19_df.Start_Lat, ".", alpha=0.5, ms=1)

plt.show()