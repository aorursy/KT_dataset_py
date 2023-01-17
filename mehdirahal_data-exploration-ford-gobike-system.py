# import all packages and set plots to be embedded inline

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sb



%matplotlib inline
df_19_06 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/201906-baywheels-tripdata.csv")

df_19_07 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/201907-baywheels-tripdata.csv")

df_19_08 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/201908-baywheels-tripdata.csv")

df_19_09 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/201909-baywheels-tripdata.csv")

df_19_10 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/201910-baywheels-tripdata.csv")

df_19_11 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/201911-baywheels-tripdata.csv")

df_19_12 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/201912-baywheels-tripdata.csv")

df_20_01 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/202001-baywheels-tripdata.csv")

df_20_02 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/202002-baywheels-tripdata.csv")

df_20_03 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/202003-baywheels-tripdata.csv")

df_20_04 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/202004-baywheels-tripdata.csv")

df_20_05 = pd.read_csv("../input/ford-gobike-system-june2019-to-may2020/data/202005-baywheels-tripdata.csv")
df_1 = pd.concat([df_19_06, df_19_07, df_19_08, df_19_09, df_19_10 , df_19_11, df_19_12, df_20_01, df_20_02, df_20_03], ignore_index = True, sort= False)

df_1.head()
df_2 = pd.concat([df_20_04, df_20_05], ignore_index = True, sort= False)

df_2.head()
df_19_06.head(1)
df_19_07.head(1)
df_19_08.head(1)
df_19_09.head(1)
df_19_10.head(1)
df_19_11.head(1)
df_19_12.head(1)
df_20_01.head(1)
df_20_02.head(1)
df_20_03.head(1)
df_20_04.head(1)
df_20_05.head(1)
df_1.shape
df_1.head()
df_1.tail()
df_1.info()
df_1.describe()
df_1.isnull().any()
df_1.duplicated().sum()
df_2.head()
df_2.tail()
df_2.duplicated().sum()
df_2.describe()
df_1_clean = df_1.copy()

df_2_clean = df_2.copy()
df_1_clean = df_1_clean.rename(columns={'duration_sec':'trip_duration', 'user_type':'member_casual', 'bike_id':'bike_ride_id', 'start_time':'started_at', \

                                       'end_time':'ended_at', 'start_station_longitude':'start_lng', 'start_station_latitude':'start_lat', \

                                        'end_station_latitude':'end_lat', 'end_station_longitude':'end_lng'})

df_2_clean = df_2_clean.rename(columns={'ride_id':'bike_ride_id'})
df_1_clean = df_1_clean.replace({'Subscriber':'member', 'Customer':'casual'})
df_2_clean.member_casual.unique()
# set df_1 datatypes

df_1_clean.started_at = df_1_clean.started_at.astype('datetime64')

df_1_clean.ended_at = df_1_clean.ended_at.astype('datetime64')

df_1_clean.start_station_id = df_1_clean.start_station_id.astype('str')

df_1_clean.end_station_id = df_1_clean.end_station_id.astype('str')

df_1_clean.bike_ride_id = df_1_clean.bike_ride_id.astype('str')

df_1_clean.member_casual = df_1_clean.member_casual.astype('category')
# set df_2 datatypes

df_2_clean.started_at = df_2_clean.started_at.astype('datetime64')

df_2_clean.ended_at = df_2_clean.ended_at.astype('datetime64')

df_2_clean.start_station_id = df_2_clean.start_station_id.astype('str')

df_2_clean.end_station_id = df_2_clean.end_station_id.astype('str')

df_2_clean.member_casual = df_2_clean.member_casual.astype('category')
df_1_clean.info()
df_2_clean.info()
df_1_clean = df_1_clean.drop(columns=['rental_access_method', 'bike_share_for_all_trip'])

df_2_clean = df_2_clean.drop(columns=['is_equity', 'rideable_type'])
print(df_1_clean.info());

print(df_2_clean.info())
df_1_clean = df_1_clean.drop_duplicates()
sum(df_1_clean.duplicated())
# trip_duration equal to end time minus started time

df_2_clean['trip_duration'] = df_2_clean['ended_at'] - df_2_clean['started_at']



# convert trip duration to second

import datetime as dt

df_2_clean['trip_duration'] = df_2_clean['trip_duration'].dt.total_seconds()



# convert trip duration datatype to int

df_2_clean['trip_duration'] = df_2_clean['trip_duration'].astype(int)
print(df_2_clean.head())

print(df_2_clean.info())
df_2_clean.describe()
df_clean = pd.concat([df_1_clean, df_2_clean])
print(df_1_clean.shape[0])

print(df_2_clean.shape[0])

print(df_clean.shape[0])
df_clean = df_clean[df_clean['trip_duration']>0]
df_clean[df_clean['trip_duration']<0].shape[0]
df_clean['trip_duration'].describe()
df_clean['trip_duration'] = df_clean['trip_duration']/60
df_clean['trip_duration'].describe()
headers = ['bike_ride_id', 'member_casual', 'started_at', 'ended_at', 'trip_duration', 'start_station_name', 'start_station_id', 'start_lat', 'start_lng', 'end_station_name', 'end_station_id', 'end_lat', 'end_lng']

df_clean = df_clean[headers]
df_clean.head()
df_clean.to_csv('ford_gobike_system.csv', index = False, header=True)

dtype = {'member_casual': 'category'}            # Set member_casual as a categorical variable

parse_dates = ['started_at', 'ended_at']         # Set date variable as a datetime datatype 

df =pd.read_csv('ford_gobike_system.csv', dtype=dtype, parse_dates=parse_dates)   # import the csv respecting datatypes
# Creating daytime, weekday and month columns



df['month'] = pd.to_datetime(df['started_at']).dt.to_period('M')

df['weekday'] = df['started_at'].dt.day_name()

df['daytime'] = df['started_at'].dt.hour



# Reorder the weekday values

df['weekday'] = pd.Categorical(df['weekday'], categories=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday', 'Sunday'],ordered=True)

df.head()
def exp_trans(x, inverse = False):

    """ transformation helper function """

    if not inverse:

        return np.log(x)

    else:

        return np.exp(x)    



data = df['trip_duration'];

bin_edges = np.arange(0, exp_trans(data.max())+0.5, 0.5);

sb.distplot(data.apply(exp_trans), bins = bin_edges);

tick_locs = np.arange(0, exp_trans(data.max())+2.5, 2.5);

plt.xticks(tick_locs, exp_trans(tick_locs, inverse = True).astype(int));

plt.xlabel("Trip Duration (exp)");





# Start with daytime distribution



base_color = sb.color_palette()[0]

sb.countplot(data=df, x='daytime', color = base_color);

plt.xlabel('Daytime')

plt.title('Daytime Distribution')
# The weekday distribution



sb.countplot(data=df, x='weekday', color=base_color);

plt.xticks(rotation = 15);

plt.xlabel('Weekday')

plt.title('Weekday Distribution')
# The month distribution



base_color = sb.color_palette()[0];

sb.countplot(data=df, x ='month', color=base_color);

plt.xticks(rotation = 45);

plt.xlabel('Month')

plt.title('Months Distribution')
# number of trips regarding the customer type



sb.countplot(data=df, x='member_casual', color=base_color);

plt.xlabel("Customer Type");

plt.title('Customer Type Count')
# make an exponential transformation for the trip duration values

df['trip_duration_exp'] = df['trip_duration'].apply(exp_trans);



# Make a function to set the yticks

def duration_yticks():

    duration_labels = ['1mn', '10mn','2h30mn', '1j', '10j'];

    duration_ticks =[1,10,150 ,1440, 14400];        ####

    plt.yticks(exp_trans(np.array(duration_ticks)), duration_labels);

    plt.ylabel("Trip Duration");
### plot the trip duration mean as a barchart

df[['member_casual', 'trip_duration']].groupby('member_casual').mean().plot(kind='bar', legend=None);

plt.xlabel("");

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean for each Customer Type');     

plt.show();

 

    



### Plot a boxplot of the trip duration regarding the customer type

sb.boxplot(data = df, x = 'member_casual', y = 'trip_duration_exp', color = base_color);                



# set y ticks and labels

duration_yticks()  



# set the tilte andx label

plt.xlabel("Customer Type");

plt.title('Trip Duration Boxplot regarding Customer Type');     



plt.show()
# Set a range from 0 to 23

hourly_ticks = np.arange(0,24,1) 



df[['daytime', 'trip_duration']].groupby('daytime').mean().plot(kind = 'bar', legend=None);

plt.xticks(hourly_ticks);

plt.ylabel('Trip Duration Mean (minute)');

plt.xlabel('Daytime');

plt.title('Trip Duration Mean for over Daytime');    
### Plot the trip duration mean regarding weeldays



df[['weekday', 'trip_duration']].groupby('weekday').mean().plot(kind='bar', legend=None);

plt.xlabel('Weekday');

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean over weekdays') 

plt.show()





### Plot the trip duration boxplot regarding weedays

sb.boxplot(data = df, x = 'weekday', y = 'trip_duration_exp', color = base_color)



# set yticks

duration_yticks()  

# title, xticks rotaion settings

plt.title('Trip Duration Boxplot regarding weekdays') 

plt.xticks(rotation = 15)

plt.show()

# Plot the trip duration mean for each months 

df[['month', 'trip_duration']].groupby('month').mean().plot(kind='bar', legend=None);

plt.xlabel('Months');

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean for each Month');     



plt.show();





# Plot boxplot of the trip duration over months 

sb.boxplot(data = df, x = 'month', y = 'trip_duration_exp', color = base_color);



# set y ticks and labels

duration_yticks();

plt.xlabel('Months');

plt.title('Trip Duration Boxplot regarding customer type');     

plt.xticks(rotation = 25);

ax = sb.countplot(data = df, x = 'daytime', hue = 'member_casual');

ax.legend(loc = 1);

plt.title('Daytime Distribution for each Customer Type');
ax = sb.countplot(data = df, x = 'weekday', hue = 'member_casual');

ax.legend(loc = 1);

plt.xticks(rotation = 15);

plt.title('Weekdays Distribution for each Customer Type');

ax = sb.countplot(data = df, x = 'month', hue = 'member_casual');

ax.legend(loc = 1);

plt.xticks(rotation = 25);

plt.title('Months Distribution for each Customer Type');

df_sub = df[['daytime', 'trip_duration', 'member_casual']].groupby(['daytime', 'member_casual']).mean();

df_sub = df_sub.add_suffix('_mean').reset_index();

sb.barplot(data = df_sub, x='daytime', y='trip_duration_mean', hue='member_casual');

plt.legend(title='Custumer Type');

plt.xlabel('Daytime');

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean over Daytime regarding Customer Type');

df_sub = df[['weekday', 'trip_duration', 'member_casual']].groupby(['weekday', 'member_casual']).mean()

df_sub = df_sub.add_suffix('_mean').reset_index()

sb.barplot(data = df_sub, x='weekday', y='trip_duration_mean', hue='member_casual')

plt.xticks(rotation=15)

plt.legend(title='Custumer Type');

plt.xlabel('Weekday');

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean over Weekdays regarding Customer Type');

df_sub = df[['month', 'trip_duration', 'member_casual']].groupby(['month', 'member_casual']).mean()

df_sub = df_sub.add_suffix('_mean').reset_index()

sb.barplot(data = df_sub, x='month', y='trip_duration_mean', hue='member_casual')

plt.xticks(rotation=25)

plt.legend(title='Custumer Type');

plt.xlabel('Month');

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean over Month regarding Customer Type');
plt.figure(figsize=(15,8));



sb.boxplot(data=df, x='weekday', y='trip_duration_exp', hue='member_casual');

# set y ticks and labels

duration_yticks();

plt.xticks(rotation = 15);

plt.legend(loc=6, bbox_to_anchor = (1.0,0.5));

plt.xlabel('Weekday');

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean over Weekdays regarding Customer Type');

plt.figure(figsize=(15,8));



sb.boxplot(data=df, x='month', y='trip_duration_exp', hue='member_casual');

# set y ticks and labels

duration_yticks();

plt.xticks(rotation = 15);

plt.legend(loc=6, bbox_to_anchor = (1.0,0.5));

plt.xlabel('Month');

plt.ylabel('Trip Duration Mean (minute)');

plt.title('Trip Duration Mean over Months regarding Customer Type');