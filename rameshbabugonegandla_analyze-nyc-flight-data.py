import pandas as pd

import numpy as np

import datetime as dt

from matplotlib import pyplot as plt 

import matplotlib

%matplotlib inline
#Read data from the given flight_data.csv file.
df_nyc_flight_data=pd.read_csv('/kaggle/input/flight_data.csv')
# Head method is used to verify sample top 5 records for the given flight_data set.

# Observations: Data consists of flight number,origin,destination,depature time, arrival time, delay in depatures,

                #delay in arrivals, air_time, 

# travel distance with date and time stamps.
df_nyc_flight_data.head(1)
# Info method is used to get a concise summary of the dataframe.

# Observations: Total record count is not matching for dep_time, dep_delay, arr_time, arr_delay, airtime
df_nyc_flight_data.info()
# Describe method is used to view some basic statistical details like percentile, mean, std etc. of a data frame of numeric values.

# Observations: Total record count is not matching for dep_time, dep_delay, arr_time, arr_delay, airtime
df_nyc_flight_data.describe()
#Identify Null values and Handle it.
df_nyc_flight_data.isnull()
#Count all NaN in a DataFrame (both columns & Rows)

#Observations: Total 46595 values were missing in the given dataset.
df_nyc_flight_data.isnull().sum().sum()
#Count total NaN at each column in DataFrame

#Observations:   a) total 8255 records not having dep_time which matches dep_delay records count.

               # b) total 8713 records not having arr_time information which we can calculate for (8713 -8255 = 458 records)

               # c) arr_delay and air_time record count is matching.

               # d) total 2512 records for tailnum is missing need to check whether we can able to fill those tailnum or not.

#Conclusions:    a) Dep_time  : 8255, Assuming 8255 flights has cancelled due to some reasons and dropped those records from analysis.

               # b) Dep_delay : 8255, Assuming 8255 flights has cancelled due to some reasons and dropped those records from analysis.

               # c) Arr_time  : 8713, (8713 - 8255 = 458) From the above point a & b, For rest of 458 records will calculate arr_time based on sched_arr_time and arr_delay.

               # d) Arr_delay : 9430, (9430 - 8255 = 1175) From the above point a & b, For rest of 1175 records will calculate arr_delay based on dep_time and arr_time.        

               # e) tailnum   : 2512, From the above point a & b, Assuming tailnum records will be 0 and having flight number details which is sufficient to proceed.

               # f) Airtime   : 9430, (9430 - 8255 = 1175) From the above point a & b, For rest of 1175 records will calculate air_time based on mean.        

        
df_nyc_flight_data.isnull().sum()
# Below task is going to perform mentioned above as conclusions.

# Analyze Dep_time and Dep_delay null values 

#Conclusions:    a) Dep_time  : 8255, Assuming 8255 flights has cancelled due to some reasons and dropped those records from analysis.

               # b) Dep_delay : 8255, Assuming 8255 flights has cancelled due to some reasons and dropped those records from analysis.



#Implementation: Used Dropna function.
df_nyc_flight_data = df_nyc_flight_data.dropna(axis=0,subset=['dep_time','dep_delay'])
df_nyc_flight_data.isnull().sum()
# Analyze arr_delay null values 

display_arr_delay_null = pd.isnull(df_nyc_flight_data["arr_delay"])

df_nyc_flight_data[display_arr_delay_null]
# Deep dive to understand one flight data which contains null values. e.g.,flight no: 464

df_nyc_flight_data_flight464 = df_nyc_flight_data[df_nyc_flight_data['flight'] == 464]

df_nyc_flight_data_flight464
# Time validation function used to calculate time. For the given dataset, if we add two time values (e.g., 1430+40 = 1470) 

# but as per time, it should be 1510 and also 2340+40 = 10 early moring hours

def time_validation(hours):

    num_hours=hours

    minutes=num_hours%100

    print(num_hours,minutes)

    if(minutes>59):

         hours=(num_hours - minutes)

         hours+=100

         #print('in if:', hours)

         if(hours>=2400):hours=hours-2400

         #print('in 2400:',hours)

         hours=hours+(minutes-60)

         #print('in hours+:',hours)

    else:

        if(hours>=2400):

            hours=hours-2400

            #print('in hours>24:',hours)

    return str(hours)

#print(time_validation(780))
# Fill all arr_time NULL values by adding sched_arr_time + dep_delay values.

arr_time_nulldata=df_nyc_flight_data[df_nyc_flight_data["arr_time"].isnull()]

arr_time_nulldata['arr_time'].fillna(arr_time_nulldata['sched_arr_time']+arr_time_nulldata['dep_delay'],inplace=True)

arr_time_nulldata['arr_time'] = arr_time_nulldata.apply(lambda row : time_validation(row['arr_time']), axis = 1) 

df_nyc_flight_data['arr_time'].fillna(value=arr_time_nulldata['arr_time'],inplace=True)

df_nyc_flight_data[df_nyc_flight_data["arr_time"].isnull()]
# No missing values for arr_time column

df_nyc_flight_data.isnull().sum()
# Fill all arr_delay NULL values by subtracting arr_time - sched_arr_time

df_nyc_flight_data['arr_time'] = pd.to_numeric(df_nyc_flight_data['arr_time'])

arr_delay_nulldata=df_nyc_flight_data[df_nyc_flight_data["arr_delay"].isnull()]

arr_delay_nulldata['arr_delay'].fillna(arr_delay_nulldata['arr_time']-arr_delay_nulldata['sched_arr_time'],inplace=True)

arr_delay_nulldata['arr_delay'] = arr_delay_nulldata.apply(lambda row : time_validation(row['arr_delay']), axis = 1) 

df_nyc_flight_data['arr_delay'].fillna(value=arr_delay_nulldata['arr_delay'],inplace=True)

df_nyc_flight_data[df_nyc_flight_data["arr_delay"].isnull()]
# No missing values for arr_delay column

df_nyc_flight_data.isnull().sum()
# Fill all air_time NULL values by subtracting arr_time - dep_time by multiplying with 65% percent of complete duration.

air_time_nulldata=df_nyc_flight_data[df_nyc_flight_data["air_time"].isnull()]

air_time_nulldata['air_time'].fillna(value=round((air_time_nulldata['arr_time']-air_time_nulldata['dep_time'])*.65),inplace=True)

air_time_nulldata['air_time'] = air_time_nulldata.apply(lambda row : time_validation(row['air_time']), axis = 1) 

df_nyc_flight_data['air_time'].fillna(value=air_time_nulldata['air_time'],inplace=True)

df_nyc_flight_data[df_nyc_flight_data["air_time"].isnull()]
# No missing values for air_time column

df_nyc_flight_data.isnull().sum()
# findday function has created to find the day name for the given date and populate it for each and every row.

import datetime 

import calendar 

  

def findDay(date): 

    full_day = datetime.datetime.strptime(date, '%d-%m-%Y').weekday() 

    return (calendar.day_name[full_day]) 



#date = '03-02-2019'

#print(findDay(date))
# flight_date column created to populate day name for each and every row.

df_nyc_flight_data['flight_date'] = df_nyc_flight_data['day'].map(str) + '-' + df_nyc_flight_data['month'].map(str) + '-' + df_nyc_flight_data['year'].map(str)

df_nyc_flight_data.head(1)
# day_name column created to populate day name for each and every row.

df_nyc_flight_data['day_name'] = df_nyc_flight_data.apply(lambda row : findDay(row['flight_date']),axis=1)

df_nyc_flight_data.head(1)
# aircraft_speed column created to populate aircraft speed for each and every row

df_nyc_flight_data['air_time']= pd.to_numeric(df_nyc_flight_data['air_time'])

aircraft_speed = df_nyc_flight_data['distance']/(df_nyc_flight_data['air_time']/60)

df_nyc_flight_data['aircraft_speed'] = aircraft_speed

df_nyc_flight_data.head(1)
# Convert dep_time,sched_dep_time,arr_time,sched_arr_time into hh:mm time format.

df_nyc_flight_data['dep_time'] = df_nyc_flight_data.dep_time[~df_nyc_flight_data.dep_time.isna()].astype(np.int64).apply('{:0>4}'.format)

df_nyc_flight_data['dep_time'] = pd.to_timedelta(df_nyc_flight_data.dep_time.str[:2]+':'+df_nyc_flight_data.dep_time.str[2:]+':00')



df_nyc_flight_data['sched_dep_time'] = df_nyc_flight_data.sched_dep_time[~df_nyc_flight_data.sched_dep_time.isna()].astype(np.int64).apply('{:0>4}'.format)

df_nyc_flight_data['sched_dep_time'] = pd.to_timedelta(df_nyc_flight_data.sched_dep_time.str[:2]+':'+df_nyc_flight_data.sched_dep_time.str[2:]+':00')



df_nyc_flight_data['arr_time'] = df_nyc_flight_data.arr_time[~df_nyc_flight_data.arr_time.isna()].astype(np.int64).apply('{:0>4}'.format)

df_nyc_flight_data['arr_time'] = pd.to_timedelta(df_nyc_flight_data.arr_time.str[:2]+':'+df_nyc_flight_data.arr_time.str[2:]+':00')



df_nyc_flight_data['sched_arr_time'] = df_nyc_flight_data.sched_arr_time[~df_nyc_flight_data.sched_arr_time.isna()].astype(np.int64).apply('{:0>4}'.format)

df_nyc_flight_data['sched_arr_time'] = pd.to_timedelta(df_nyc_flight_data.sched_arr_time.str[:2]+':'+df_nyc_flight_data.sched_arr_time.str[2:]+':00')



df_nyc_flight_data.head(1)
# Created two new columns dep_status and arr_status.

# dep_status column used to store information based on the flight depatured before_ontime, ontime, Dep_Actualdelay information.

# arr_status column used to store information based on the flight depatured before_ontime, ontime, Arr_Actualdelay information.

df_nyc_flight_data.loc[df_nyc_flight_data.dep_delay < 0, "dep_status"]="Before_OnTime"

df_nyc_flight_data.loc[df_nyc_flight_data.dep_delay == 0, "dep_status"]="OnTime"

df_nyc_flight_data.loc[df_nyc_flight_data.dep_delay > 0, "dep_status"]="Dep_ActualDelay"

df_nyc_flight_data['arr_delay'] = pd.to_numeric(df_nyc_flight_data['arr_delay'])

df_nyc_flight_data.loc[df_nyc_flight_data.arr_delay < 0, "arr_status"]="Before_OnTime"

df_nyc_flight_data.loc[df_nyc_flight_data.arr_delay == 0, "arr_status"]="OnTime"

df_nyc_flight_data.loc[df_nyc_flight_data.arr_delay > 0, "arr_status"]="Arr_ActualDelay"



df_nyc_flight_data.head(1)
#Created one new column quarter to fill quarter values from time_hour date column

#Convert datatypes into datetime for the required columns

df_nyc_flight_data['flight_date']= pd.to_datetime(df_nyc_flight_data['flight_date']) 

df_nyc_flight_data['time_hour']= pd.to_datetime(df_nyc_flight_data['time_hour'])

df_nyc_flight_data['quarter'] = df_nyc_flight_data['time_hour'].dt.quarter
 #Convert datatypes into category for the required columns

df_nyc_flight_data[['month','day', 'carrier', 'origin', 'dest', 'day_name','dep_status','arr_status','quarter']] = df_nyc_flight_data[['month','day', 'carrier', 'origin', 'dest', 'day_name','dep_status','arr_status','quarter']].apply(lambda x: x.astype('category'))
df_nyc_flight_data.head(1)
#Verify all columns datatypes and convert it as per the data visualization requirement.

df_nyc_flight_data.dtypes
#Verify whether we have any missingvalues for all columns

df_nyc_flight_data.isnull().sum()
# To verify maximum aircraft_speed for the individual carrier.

# Observation: Interestingly found 'inf' value as maximum for carrier 'MQ'

carrier_speed = df_nyc_flight_data.groupby(['carrier'])['aircraft_speed'].max()

carrier_speed
# Observation: Interestingly found 'inf' value as maximum for carrier 'MQ'

# Interesting Facts: Noticied that, Dep_time and Arr_time for this flight '3678' is same and which should not be the case

# in real-time.

df_nyc_flight_data.sort_values(by='aircraft_speed',ascending=False).head(2)
# Observation: Interestingly found 'inf' value as maximum for carrier 'MQ'

# Interesting Facts: Noticied that, Dep_time and Arr_time for this flight '3678' is same and which should not be the case

# in real-time.

#df_nyc_flight_data_flight = df_nyc_flight_data[df_nyc_flight_data['flight'] == 3678]

df_nyc_flight_data_flight = df_nyc_flight_data[df_nyc_flight_data['air_time'] == 0]

df_nyc_flight_data_flight.head(2)

# Observation: Interestingly found 'inf' value as maximum for carrier 'MQ'

# Interesting Facts: Noticied that, Dep_time and Arr_time for this flight '3678' is same and which should not be the case

# in real-time.

# Conclusion:Dropped this flight so that, can able to analyze problem statement 'Aircraft speed analysis' for all carriers.

df_nyc_flight_data = df_nyc_flight_data.drop([259244]) 

df_nyc_flight_data_flight = df_nyc_flight_data[df_nyc_flight_data['flight'] == 0]

df_nyc_flight_data_flight.head(2)

#df_nyc_flight_data[df_nyc_flight_data['aircraft_speed'].isin([np.inf, -np.inf])]

#df_nyc_flight_data.at[259244,'aircraft_speed']=325.83 # mean value
#Verify whether data displays values correct or not.

carrier_speed = df_nyc_flight_data.groupby(['carrier'])['aircraft_speed'].max()

carrier_speed
# Calculate minimum and maximum dep_delay values for the given NYC flight data set.

min_dep_delay = min(df_nyc_flight_data.dep_delay)

print(min_dep_delay)

max_dep_delay = max(df_nyc_flight_data.dep_delay)

print(max_dep_delay)
# Visualizations Heading: Identify Departure Delay information based on Origin and Carrier

# Plot Used   : Relational Plot

# Description : Fetched all records whose dep_delay > 0 and plotted graph based on Origin and Carrier

# Outcome     : a) Carrier HA, Origin JFK, is the one whose maximum dep_delay is high > 1200

#               b) Carrier OO, Origin LGA, is the one whose dep_delay is very less.

#               c) Carrier AS, Origin EWR, is the one whose dep_delay is very less.

import seaborn as sns

df_nyc_flight_data_dep_actualdelay=df_nyc_flight_data[df_nyc_flight_data["dep_status"]=="Dep_ActualDelay"]

sns.relplot(x="carrier", y="dep_delay", hue="origin", data=df_nyc_flight_data_dep_actualdelay);
# Visualizations Heading: Identify Depature Delay information based on Origin and Day_name

# Plot Used   : Categorical Plot

# Description : Fetched all records whose dep_delay > 0 and plotted graph based on Origin and Day_name.

# Outcome     : a) Day: Wednesday, Origin JFK, is the one whose maximum dep_delay is around > 1200

#               b) Day: Saturday, Origin JFK, is the second whose maxmimum dep_delay is around > 1100

#               c) Day: Tuesday, Less number of delays on this day.

sns.catplot(x="day_name", y="dep_delay", hue="origin", jitter=False, aspect=2,data=df_nyc_flight_data_dep_actualdelay);
# Calculate total counts of flight depature delay based on dep_status == Dep_ActualDelay

depDelay_count = df_nyc_flight_data[df_nyc_flight_data["dep_status"]=="Dep_ActualDelay"]

depDelay_count['dep_status'].value_counts()
# Due to huge data set, Calculate top 2500 flight depature delay based on dep_status == Dep_ActualDelay by descending order

Top2500_DepDelays = depDelay_count.sort_values(by='dep_delay',ascending=False).head(2500)

Top2500_DepDelays['dep_status'].value_counts()
# Visualizations Heading: Identify Top 2500 Depature Delay information based on Origin and Carrier

# Plot Used   : Relational Plot

# Description : Fetched all records whose dep_delay > 0 and plotted graph based on Origin and Carrier

# Outcome     : a) Carrier HA, Origin JFK, is the one whose maximum dep_delay is around > 1200

#               b) Carrier MQ, Origin JFK, is the second whose maxmimum dep_delay is around > 1100.

#               c) Carrier AS, Origin EWR, is the one whose dep_delay is very less.

sns.relplot(x="carrier", y="dep_delay", hue="origin", data=Top2500_DepDelays,aspect=2);
# Visualizations Heading: Identify Top2500 Depature Delay information based on Origin and Day_name

# Plot Used   : Categorical Plot

# Description : Fetched all records whose dep_delay > 0 and plotted graph based on Origin and Day_name.

# Outcome     : a) Day: Wednesday, Origin JFK, is the one whose maximum dep_delay is around > 1200

#               b) Day: Saturday, Origin JFK, is the second whose maxmimum dep_delay is around > 1100

#               c) Day: Tuesday, Less number of delays on this day.

sns.catplot(x="day_name", y="dep_delay", hue="origin", jitter=False, aspect=2,data=Top2500_DepDelays);
# Calculate total flight counts, Percentage based on Origin and dep_status = OnTime

dep_OnTime = df_nyc_flight_data.groupby('origin')['origin'].count().reset_index(name='total')

OnTimeFlights = df_nyc_flight_data.loc[df_nyc_flight_data['dep_status']=='OnTime'].groupby(['origin','dep_status'])['dep_status'].count().unstack('dep_status')

dep_OnTime['OnTime'] = OnTimeFlights['OnTime'].values 

dep_OnTime['percentage'] = (dep_OnTime['OnTime']/dep_OnTime['total'])*100

dep_OnTime
# Calculate total flight counts, Percentage based on Origin and dep_status = Before_OnTime

dep_Before_OnTime = df_nyc_flight_data.groupby('origin')['origin'].count().reset_index(name='total')

Before_OnTimeFlights = df_nyc_flight_data.loc[df_nyc_flight_data['dep_status'] == 'Before_OnTime'].groupby(['origin','dep_status'])['dep_status'].count().unstack('dep_status')

dep_Before_OnTime['Before_OnTime'] = Before_OnTimeFlights['Before_OnTime'].values

dep_Before_OnTime['percentage'] = (dep_Before_OnTime['Before_OnTime']/dep_Before_OnTime['total'])*100

dep_Before_OnTime
# Calculate total flight counts, Percentage based on Origin and dep_status = Dep_Actualdelay

dep_ActualDelay = df_nyc_flight_data.groupby('origin')['origin'].count().reset_index(name='total')

ActualDelay_Flights = df_nyc_flight_data.loc[df_nyc_flight_data['dep_status'] == 'Dep_ActualDelay'].groupby(['origin','dep_status'])['dep_status'].count().unstack('dep_status')

dep_ActualDelay['Dep_ActualDelay'] = ActualDelay_Flights['Dep_ActualDelay'].values

dep_ActualDelay['percentage'] = (dep_ActualDelay['Dep_ActualDelay']/dep_ActualDelay['total'])*100

dep_ActualDelay
# Merge above all 3 dataframes and display total flight counts, Percentage based on Origin and dep_status in (<0,==0,>0)

merged_inner1 = pd.merge(left=dep_OnTime, right=dep_Before_OnTime, left_on='origin', right_on='origin')

merged_inner1.shape

merged_inner_final = pd.merge(left=merged_inner1, right=dep_ActualDelay, left_on='origin', right_on='origin')

merged_inner_final.shape

merged_inner_final
# Rename columns with meaning full names for the merged dataframe and display final result set.

merged_inner_final = merged_inner_final.drop(['total_y', 'total'], axis = 1) 

merged_inner_final.rename(columns = {'total_x':'total'}, inplace = True) 

merged_inner_final.rename(columns = {'percentage_x':'OnTime_percentage'}, inplace = True) 

merged_inner_final.rename(columns = {'percentage_y':'BeforeOnTime_percentage'}, inplace = True) 

merged_inner_final.rename(columns = {'percentage':'ActualDepDelay_percentage'}, inplace = True) 

merged_inner_final
# Visualizations Heading: Identify Best origin airports on basis of time departure Percentage.

# Plot Used   : Pie Plot

# Description : Pie Chart plotted based on time departure from the origin.

# Outcome     : a) Pie 1: OnTime Flights Percentage based on Origin. Origin JFK is highest percentage with 37.83%

#               b) Pie 2: Before_OnTime Flights Percentage based on Origin. Origin LGA is highest percentage with 36.91%

#               c) Pie 3: ActualDepDelay Flights Percentage based on Origin. Origin EWR is highest percentage with 38.50%

#               d) Pie 4: Total number of Flights Percentage based on Origin. Origin EWR is highest percentage with 35.80%

# Make figure and axes

fig, axs = plt.subplots(2, 2,figsize=(10,5))



# A standard pie plot

axs[0,0].set_title('OnTime Flights Percentage based on Origin')

axs[0,0].pie(merged_inner_final['OnTime_percentage'], labels=merged_inner_final['origin'], autopct='%1.2f%%', shadow=True,explode=(0.1, 0, 0))



axs[0,1].set_title('Before_OnTime Flights Percentage based on Origin')

axs[0,1].pie(merged_inner_final['BeforeOnTime_percentage'], labels=merged_inner_final['origin'], autopct='%1.2f%%', shadow=True,explode=(0.1, 0, 0))



axs[1,0].set_title('ActualDepDelay Flights Percentage based on Origin')

axs[1,0].pie(merged_inner_final['ActualDepDelay_percentage'], labels=merged_inner_final['origin'], autopct='%1.2f%%', shadow=True,explode=(0.1, 0, 0))



axs[1,1].set_title('Total number of Flights Percentage based on Origin')

axs[1, 1].pie(merged_inner_final['total'], labels=merged_inner_final['origin'], autopct='%1.2f%%', shadow=True,explode=(0.1, 0, 0))



plt.show()
# Visualizations Heading: Identify TotalNumberofFlights departure from Origin (i.e,Best Airports.)

# Plot Used   : Bar Plot

# Description : Bar Chart plotted to show total number of flights departure from the origin.

# Outcome     : a) Origin JFK is one of the highest number OnTime Flights timely departure 

#               b) Origin LGA is one of the highest number Before_OnTime Flights timely departure

#               c) Origin EWR is one of the highest number Dep_ActualDelay Flights timely departure

import numpy as np



x = np.arange(len(merged_inner_final['origin']))  # the label locations

width = 0.25  # the width of the bars



fig, ax = plt.subplots(figsize=(10,5))

rects1 = ax.bar(x, merged_inner_final['OnTime'], width,alpha=0.5,color='#EE3224', label='OnTime')

rects2 = ax.bar([p + width for p in x], merged_inner_final['Before_OnTime'], width,alpha=0.5, color='#F78F1E',label='Before_OnTime')

rects3 = ax.bar([p + width*2 for p in x], merged_inner_final['Dep_ActualDelay'], width, alpha=0.5,color='#FFC222',label='Dep_ActualDelay')



# Add some text for labels, title and custom x-axis tick labels, etc.

ax.set_ylabel('Total Number of Flights')

ax.set_title('TotalNumberofFlights departure from Origin i.e,Best Airports')

ax.set_xticks([p + 1.5 * width for p in x])

ax.set_xticklabels(merged_inner_final['origin'])

ax.legend()



plt.xlim(min(x)-width, max(x)+width*4)

plt.ylim([0, max(merged_inner_final['OnTime'] + merged_inner_final['Before_OnTime'] + merged_inner_final['Dep_ActualDelay'])] )



fig.tight_layout()

plt.grid()

plt.show()

# Visualizations Heading: Identify TotalNumberofFlights departure from Origin (i.e,Best Airports.)

# Plot Used   : line Plot

# Description : line Chart plotted to show total number of flights departure from the origin.

# Outcome     : a) Origin EWR,JFK,LGA is having similar number (slightly differ) OnTime Flights timely departure 

#               b) Before_OnTime Flights  timely departure count decreases from EWR,JFK,LGA

#               c) Dep_ActualDelay Flights timely departure count increases from EWR,JFK,LGA

ax = plt.gca()



merged_inner_final.plot(kind='line',x='origin',y='OnTime',color='#EE3224', ax=ax)

merged_inner_final.plot(kind='line',x='origin',y='Before_OnTime', color='#F78F1E', ax=ax)

merged_inner_final.plot(kind='line',x='origin',y='Dep_ActualDelay', color='#FFC222', ax=ax)

plt.show()
# Visualizations Heading: Identify Aircraft_Speed based on airtime, distance and Origin. 

# Plot Used   : Pair Plot

# Description : Pair Plot plotted to show how the aircraft_speed increases based on air_time and distance.

# Outcome     : a) Origin LGA is having high aircraft_speed which travels shorter distance.

#               b) Origin LGA is having consistent aircraft_speed for distance <1800

#               c) Origin EWR is having inconsistent aircraft_speed for distance <2500

import seaborn as sns

origin_speed = sns.pairplot(df_nyc_flight_data,height = 3,vars = ['distance','air_time','aircraft_speed'],hue='origin',palette="husl",markers=["o", "s", "D"])

plt.show(origin_speed)
# Calculate mean of aircraft_speed by applying group by on carrier.

carrier_speed = df_nyc_flight_data.groupby(['carrier'])['aircraft_speed'].mean()

carrier_speed
# Calculate mean of distance by applying group by on carrier.

carrier_distance = df_nyc_flight_data.groupby(['carrier'])['distance'].mean()

carrier_distance
# Merge above two dataframes to display records for Carrier, aircraft_speed and distance.

merged_inner = pd.merge(left=carrier_speed, right=carrier_distance, left_on='carrier', right_on='carrier')

merged_inner.shape

merged_inner.info()
# reset index for the above dataframe records.

merged_inner.reset_index(level=0,drop=False,inplace=True)

merged_inner.head(1)
# Visualizations Heading: Identify Aircraft_Speed based on distance and Origin. 

# Plot Used   : Scatter Plot

# Description : Scatter Plot plotted to show how the aircraft_speed increases based on distance for carrier.

# Outcome     : a) Carrier HA is having high aircraft_speed for the distance covered around 5000

#               b) Carrier YU is having low aircraft_speed for the distance covered around 400

#               c) Most of the Carrier is having consistent aircraft_speed for the similar distance covered.

f, ax = plt.subplots(figsize=(6.5, 6.5))

sns.set(style="ticks")

sns.despine(f, left=True, bottom=True)

sns.scatterplot(x="distance", y="aircraft_speed",

                hue="carrier", 

                palette="dark",

                sizes=(1, 8), linewidth=0,

                data=merged_inner, ax=ax)

#Calculate count of OnTime,Before_OnTime,Arr_ActualDelay arrival count by grouping with arr_status column

arr_Status = df_nyc_flight_data.groupby('arr_status')['arr_status'].count()

print('arr_Status complete list:')

print(arr_Status)
# Visualizations Heading: Visualize the Ontime arrival percentage and count.

# Plot Used   : Count Plot and Pie Plot

# Description : Count Plot plotted to show the count of number of flights based on OnTime,Before_OnTime,Arr_ActualDelay arrival status.

#             : Pie Plot plotted to show the percentage of number of flights arrivals based on OnTime,Before_OnTime,Arr_ActualDelay arrival status.

# Outcome     : a) Around 189038 number of flights has arrived Before_OnTime with 57.54%

#               b) Around 134057 number of flights has arrived Arr_ActualDelay with 40.81%

#               c) Around 5425 number of flights has arrived OnTime with 1.65%

# Setting up the chart area

f,ax=plt.subplots(1,2,figsize=(14,7))



# setting up chart 

df_nyc_flight_data['arr_status'].value_counts().plot.pie(explode=[0,0,0.2], autopct='%1.2f%%',ax=ax[1], shadow=False)   

 

# setting title for pei chart

ax[1].set_title('On time arrival % Analysis')

ax[1].set_ylabel('')



# setting up data for barchart

sns.countplot('arr_status',order = df_nyc_flight_data['arr_status'].value_counts().index, data=df_nyc_flight_data,ax=ax[0])

ax[0].set_title('Arrival Status of total flights (in numbers)')

ax[0].set_ylabel('Number of Flights')



plt.show()

# Calculate maximum number of flights to particular destination.

df_nyc_flight_data['dest'].value_counts()

# Calculate maximum number of flights to started from particular origin.

df_nyc_flight_data['origin'].value_counts()
# Calculate Top 25 Maximum number of flights from origin to destination by applying group by on origin and dest.

Top25_MaxFlights_Dest = df_nyc_flight_data.groupby('origin')['dest'].value_counts().to_frame()

#Top25_MaxFlights_Dest.dtypes

Top25_MaxFlights_Dest.rename(columns = {'dest':'dest_count'}, inplace = True) 

Top25_MaxFlights_Dest.reset_index(level=1,drop=False,inplace=True)

Top25_MaxFlights_Dest.reset_index(level=0,drop=False,inplace=True)

#sum_flight1.head(40)

Top25_MaxFlights_Dest = Top25_MaxFlights_Dest.sort_values(by='dest_count',ascending=False).head(25)

Top25_MaxFlights_Dest.reset_index(level=0,drop=False,inplace=True)

Top25_MaxFlights_Dest.head(3)
# Visualizations Heading: Visualize Top 25 Maximum number of flights headed to some particular destination.

# Plot Used   : Relational Plot

# Description : Relational Plot plotted to show Top 25 Maximum number of flights headed to some particular destination.

# Outcome     : a) Top 1  : Origin JFK is having maximum number of flights headed towards LAX destination

#               b) Top 10 : Origin EWR is having maximum number of flights headed towards BOS destination

#               c) Top 25 : Origin EWR is having maximum number of flights headed towards FLL destination

palette = sns.cubehelix_palette(light=.8, n_colors=3)

sns.relplot(x="dest", y="dest_count", hue="origin", height=6, aspect=3, data=Top25_MaxFlights_Dest);
# Visualizations Heading: Visualize Top 25 Maximum number of flights headed to some particular destination.

# Plot Used   : Catergorical Plot

# Description : Catergorical Plot plotted to show Top 25 Maximum number of flights headed to some particular destination.

# Outcome     : a) Top 1  : Origin JFK is having maximum number of flights headed towards LAX destination

#               b) Top 10 : Origin EWR is having maximum number of flights headed towards BOS destination

#               c) Top 25 : Origin EWR is having maximum number of flights headed towards FLL destination

g = sns.catplot(x="dest", y="dest_count", hue="origin", data=Top25_MaxFlights_Dest,

                height=5, kind="bar", palette="muted", aspect=4)

g.despine(left=True)

g.set_ylabels("Total Number of Flights")
# Visualizations Heading: Visualize  Month-Wise mean analysis about the Flight Departure and Arrival Status.

# Plot Used   : line Plot

# Description : line Plot plotted to show month-wise analysis  dep_delay and arr_delay .

# Outcome     : a) left-side line plot shows 3,5,9 month is having earlier arrivals and 4,9 month is having earlier departure.

#               b) right-side line plot shows 4,6,7 month is having arrivals delays and 4,6,7 month is having departure delays.



f,ax=plt.subplots(1,2,figsize=(20,8))



dep_Ontime = df_nyc_flight_data[df_nyc_flight_data["dep_status"]!="Dep_ActualDelay"]

dep_Ontime[['month','arr_delay','dep_delay']].groupby(['month']).mean().plot(ax=ax[0],marker='*',linestyle='dashed',color = 'b'+'r',linewidth=2, markersize=12)



df_nyc_flight_data_dep_actualdelay[['month','arr_delay','dep_delay']].groupby(['month']).mean().plot(ax=ax[1],marker='*',linestyle='dashed',color = 'b'+'r',linewidth=2, markersize=12)



# Visualizations Heading: Visualize Quarter-Wise analysis of Flight mean Depature Delays by Carriers.

# Plot Used   : Heat Map

# Description : Heat Map plotted to show  Quarter-Wise analysis of Flight mean Depature Delays by Carriers.

# Outcome     : a) Quarter 1: Carrier EV is having highest (24.3) & Carrier US is having lowest (2.7) mean depature delay.

#               b) Quarter 2: Carrier F9 is having highest (26.6) & Carrier HA is having lowest (0.3) mean depature delay.

#               c) Quarter 3: Carrier FL is having highest (21.9) & Carrier AS is having lowest (5.1) mean depature delay.

#               d) Quarter 4: Carrier FL is having highest (21.7) & Carrier OO is having lowest (0.8) mean depature delay.

import seaborn as sns

plt.figure(figsize=(18,10))

plt.title("Quarter-Wise flight depature delays by carriers")

plt.tight_layout()

hmap = pd.pivot_table(df_nyc_flight_data,values='dep_delay',aggfunc='mean',index='carrier',columns='quarter')

sns.heatmap(hmap,annot=True,cmap="YlGnBu",center=0,linewidths=.2,fmt='g')

plt.show()