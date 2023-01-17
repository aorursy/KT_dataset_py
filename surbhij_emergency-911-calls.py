# Importing the required libraries:

import pandas as pd
import datetime
import collections
import seaborn as sns
import matplotlib.pyplot
import numpy as np
# Identifying the busiest year for 911 Emergency calls:

emergency=pd.read_csv('../input/911.csv')
emergency.head()
a=emergency['timeStamp'].tolist()
year=[]
for i in range(0,len(a)):
    date = datetime.datetime.strptime(a[i], "%Y-%m-%d %H:%M:%S")
    year.append(date.year)
year
ctr = collections.Counter(year)
ye=[2015,2016,2017,2018]
freq=[ctr[2015],ctr[2016],ctr[2017],ctr[2018]]
d={'Year':ye,'No. of calls recorded':freq}
df=pd.DataFrame(data=d)
df
sns.barplot(x='Year',y='No. of calls recorded',data=df)

# INSIGHT:
# It is observed, from the following barplot, that 2016 was the most busiest year as almost 1,40,000 emergency calls 
# were recorded followed by year 2017.

# Identifying the busiest month for 911 Emergency calls: 
month=[]
for i in range(0,len(a)):
    date = datetime.datetime.strptime(a[i], "%Y-%m-%d %H:%M:%S")
    month.append(date.month)
month
ctr_mon = collections.Counter(month)
month1=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
calls=[ctr_mon[1],ctr_mon[2],ctr_mon[3],ctr_mon[4],ctr_mon[5],ctr_mon[6],ctr_mon[7],ctr_mon[8],ctr_mon[9],ctr_mon[10],ctr_mon[11],ctr_mon[12]]
d1={'Month':month1,'No. of calls recorded':calls}
df1=pd.DataFrame(data=d1)
sns.barplot(x='Month',y='No. of calls recorded',data=df1)

# INSIGHT:
# It is clearly observable from the following barplot that no. of calls recorded for month 4 to 11 are less or more similar 
# while for rest of month the no. are quite large compared to previous ones.
# The highest no. of calls are recorded for month 'January' followed by 'March' and the lowest no. of calls recorded for 
# month 'April'.
# 'January' seems to be the busiest month.
# Checking the number of calls in each year month wise:

year_month=pd.DataFrame({'Year':year,'Month':month})
yearwise_freq_of_months=year_month.groupby(['Year','Month']).size().unstack()
yearwise_freq_of_months.columns=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
mon=list(yearwise_freq_of_months.columns.values)
yearwise_freq_of_months.plot(kind='bar',figsize=(10,10))

# INSIGHT:
# The above observation that January is the busiest month, i.e. most number of emergency calls were recorded in the 
# month of January, can also be visualised from the following barplot where we can see that maximum number of calls in
#  year 2016 and 2018 were recorded in the month of January.
# Identifying the Top 6 towns in terms of maximum no. of calls recorded:
town_freq=pd.Series(emergency['twp'])
town_freq.value_counts(ascending=False)[0:6]

# INSIGHT:
# 'Lower Merion, 'Abington','Norristown','Upper Merion', 'Cheltenham' and 'Pottstown are the Top-6 towns 
# in which most of the emergency calls were recorded. 
# Also, 112517 calls out of 326425 calls were recorded for these towns, which is nearly 35% of entire data.
# Identifying the Type of cases(Accident type) for which 911 has been called for the maximum number of times:

type_of_acc=emergency['title']
type_of_acc=type_of_acc.tolist()
acc=[]
final_acc=[]
for i in range(0,len(type_of_acc)):
    acc.append(type_of_acc[i].split(':'))
    final_acc.append(acc[i][0])
final_acc
ctr_acc = collections.Counter(final_acc)
ctr_acc
title=['EMS','Fire','Traffic']
accidents=[ctr_acc['EMS'],ctr_acc['Fire'],ctr_acc['Traffic']]
d2={'Accident Type':title,'No. of calls recorded':accidents}
df2=pd.DataFrame(data=d2)
df2
sns.barplot(x="Accident Type",y="No. of calls recorded",data=df2)

# INSIGHT:
# It is observed that maximum no. of 911 emergency calls were recorded for the 'EMS' followed by 'Traffic'.
# Finding the number of cases of EMS, Fire and Traffic accidents in each of the years:

year_acc=pd.DataFrame({'Year':year,'Accident Type':final_acc})
yearwise_freq_of_acc=year_acc.groupby(['Year','Accident Type']).size().unstack()
yearwise_freq_of_acc.plot(kind='bar',figsize=(10,10))

# INSIGHT:
# The above claim/insight drawn, that most of the calls to 911 are for EMS, is verified from the plot below which shows
# that the number of calls to 911 for EMS cases are greater than the rest of two cases in all the four years.
# Identifying the Time of the day in which most of the emergency calls were recorded:

hour=[]
for i in range(0,len(a)):
    date = datetime.datetime.strptime(a[i], "%Y-%m-%d %H:%M:%S")
    hour.append(date.hour)
hour
hour_desc=[]
for i in range(0,len(hour)):
    if(hour[i]>0 and hour[i]<5):
        hour_desc.append('After Midnight')
    elif(hour[i]>4 and hour[i]<12):
        hour_desc.append('Morning')
    elif(hour[i]>11 and hour[i]<17):
        hour_desc.append('Afternoon')
    elif(hour[i]>17 and hour[i]<21):
        hour_desc.append('Evening')
    else:
        hour_desc.append('Night')
ctr_hour = collections.Counter(hour_desc)
time_slot=['After Midnight','Morning','Afternoon', 'Evening','Night']
no_of_calls=[ctr_hour['After Midnight'],ctr_hour['Morning'],ctr_hour['Afternoon'],ctr_hour['Evening'],ctr_hour['Night']]
d3={'Time of the day':time_slot,'No. of calls recorded':no_of_calls}
time_df=pd.DataFrame(data=d3)
time_df
sns.barplot(x="Time of the day",y="No. of calls recorded",data=time_df,palette='Blues')

# INSIGHT:
# From the following plot, we can easily conclude that the 'Afternoon' time of the day (i.e from 12:00 to 16:00) is the
# busiest time for 911 Emergency calls as they recieved the maximum number of calls in this time slot followed by 'Morning'
# time of the day.
# Also, minimum no. of emergency calls were recorded for 'After Midnight'. 
# Plotting the no. of calls recorded for each month according to Time of the day:

month_hour=pd.DataFrame({'Month':month,'Time of the Day':hour_desc})
monthwise_freq_of_hours=month_hour.groupby(['Month','Time of the Day']).size().unstack()
monthwise_freq_of_hours.plot(kind='bar',figsize=(10,10))
matplotlib.pyplot.xticks(np.arange(12), ('Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'))

# INSIGHT:
# From the following plot, we can see that 'Morning' time is the busiest time for 911 Emergency calls for all the months  
# followed by 'Afternoon' time of the day.
# Identifying the Busiest Station based on the no. of 911 Emergency calls recorded:

desc=emergency['desc'].tolist()
desc[0:5]
new_desc=[]
for i in range(0,len(desc)):
    new_desc.append(desc[i].split(';'))
new_desc[1][2]
station=[]
for i in range (0,len(new_desc)):
    for j in new_desc[i]:
        if 'Station' in j:
            station.append(j) 
station
station_new=[]
for i in station:
    station_new.append(i.split('Station')[-1])
station_new
station_final=[]
for i in station_new:
    station_final.append(i.split(':')[-1])
station_final
len(station_final)
busy_station=pd.Series(station_final)
busy_station.value_counts(ascending=False)[0:6]

# INSIGHT:
# From the results obtained, we can conclude that the Station '308A' is the busiest one (12352 calls were recorded)
# followed by '329' and '313', where more than 10,000 calls were recorded in these 4 years.
# The minimum no. of calls(7137 calls) were recorded for Station '345'.
# Identifying the Top-12 set of lattitude and longitude in which maximum no. of calls were recorded:
new_lat=round(emergency['lat'],2)
new_lon=round(emergency['lng'],2)

lat_lon = pd.DataFrame({'lat':emergency['lat'].tolist(),'lon':emergency['lng'].tolist()})
count_lat_lon = lat_lon.groupby(['lat','lon'])['lon'].size()
count_lat_lon_df = count_lat_lon.to_frame()
count_lat_lon_df.sort_values(by='lon',ascending=False)
busy_lat_lon = count_lat_lon_df.sort_values(by='lon',ascending=False)[0:12]
busy_lat_lon

# From the result obtained, we observed that more than 1000 calls were recorded for the Top-11 lattitude and longitude 
# set.
# For lattitude: 40.097222, longitude: -75.376195, 4718 Emergency calls were recorded.