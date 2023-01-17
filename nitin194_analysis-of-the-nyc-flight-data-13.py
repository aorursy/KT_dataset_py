your_local_path="../input/"
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import warnings; warnings.simplefilter('ignore')
from IPython.display import Markdown, display

def printmd(string):
    display(Markdown(string))
data = pd.read_csv(your_local_path+"flight_data.csv")
data.head()
missing_data = data.isnull().sum()

total_cells = np.product(data.shape)
print("Total number of cells in dataset: ",total_cells)

missing_cells = missing_data.sum()
print("Total number of missing cells in dataset: ",missing_cells)

percent_miss = (missing_cells/total_cells)*100
print("Total percentage for missing column is approx: {:1.1}%".format(percent_miss))
data[data.isnull().any(axis=1)]

import seaborn as sns
print(data.dep_delay.describe())
print("\n***Top 5 frequently occuring values in the column***")
print(data.dep_delay.value_counts().head(5)) #top 5 frequently occuring values in the column
print("\n***Boxplot to visualize the data distribution***")
sns.boxplot(y='dep_delay', data=data)
plt.ylim(-40,50)
plt.show()

printmd("*Above data shows that dep_delay data is highly dispersed with lot many outliers\
        and mean value lying just above the 75% quartile. Also, we can see a range of frequently\
        occuring values which are negative integers.But we will use average value to fill the missing\
        values in the respective column to sustain the column average or it will deviate.*")
#fills_list = data.dep_delay.value_counts().index[:5]
#data['dep_delay'] = data['dep_delay'].map(lambda x: np.random.choice(fills_list) if np.isnan(x) else x)
average = data.dep_delay.mean()
data.dep_delay.fillna(average, inplace=True)
data.dep_delay = data.dep_delay.round()
data.dep_time = np.nan_to_num(data.dep_time).astype(int) ##Convert all values to Integer while converting NAN values to 0.
data.dep_time = data.dep_time.map(lambda x: 9999 if x == 0 else x ) ##Convert all 0 values to 9999
data.dep_time = data.dep_time.map(lambda x: 0 if x == 2400 else x ) ##Convert all 2400 time values to 0000
data.dep_time = data.dep_time.map("{:04}".format) ##Convert all time values to fixed 4 digit integer by appending leading 0's
data.dep_time = pd.to_datetime(data[data.dep_time != '9999'].dep_time, format='%H%M').dt.time ##Convert 4 digit integer value to time format in HH:MM:SS
data.sched_dep_time = data.sched_dep_time.map("{:04}".format) ##Similar as above
data.sched_dep_time = pd.to_datetime(data.sched_dep_time, format='%H%M').dt.time 
for i in data[data.dep_time.isnull()].index.tolist():
    original = data.iloc[i,4]
    delay = data.iloc[i,5]
    t2 = dt.timedelta(minutes=delay) 
    data.iloc[i,3] = (dt.datetime.combine(dt.date(1,1,2),original) + t2).time()
data.arr_time = np.nan_to_num(data.arr_time).astype(int) ##Similar as above
data.arr_time = data.arr_time.map(lambda x: 9999 if x == 0 else x )
data.arr_time = data.arr_time.map(lambda x: 0 if x == 2400 else x )
data.arr_time = data.arr_time.map("{:04}".format)
data.arr_time = pd.to_datetime(data[data.arr_time != '9999'].arr_time, format='%H%M').dt.time
data.sched_arr_time = data.sched_arr_time.map("{:04}".format)
data.sched_arr_time = pd.to_datetime(data.sched_arr_time, format='%H%M').dt.time
for i in data[data.arr_time.notnull()][data.arr_delay.isnull()].index.tolist():
    original=data.iloc[i,7]
    actual=data.iloc[i,6]
    t1 = dt.timedelta(hours=original.hour,minutes=original.minute)
    t2 = dt.timedelta(hours=actual.hour,minutes=actual.minute)
    data.iloc[i,8]=(t2-t1).seconds/60
print(data.arr_delay.describe())
print("\n***Top 10 frequently occuring values in the column***")
print(data.arr_delay.value_counts().head(10))
print("\n***Boxplot to visualize the data distribution***")
sns.boxplot(y='arr_delay', data=data, linewidth=1)
plt.ylim(-100,100)
plt.show()
printmd("*Above data shows that dep_delay data is also highly dispersed with lot many outliers\
        and mean value lying below the 75% quartile. Also, we can see a range of frequently\
        occuring values which are negative integers.But we will use average value to fill the missing\
        values in the respective column to sustain the column average or it will deviate and sometimes I observed\
        that filling NaN values with frequent negative integers may result in divide by 0 while calculating air time.*")
#fill_list = data.arr_delay.value_counts().index[:10]
#data['arr_delay'] = data['arr_delay'].map(lambda x: np.random.choice(fill_list) if np.isnan(x) else x)
average_delay = data.arr_delay.mean()
data.arr_delay.fillna(average_delay, inplace=True)
data.arr_delay = data.arr_delay.round()
for i in data[data.arr_time.isnull()].index.tolist():
    original = data.iloc[i,7]
    delay = data.iloc[i,8]
    t2 = dt.timedelta(minutes=delay) 
    data.iloc[i,6] = (dt.datetime.combine(dt.date(1,1,2),original) + t2).time() #dt.datetime.combine(...) lifts the datetime.time to a datetime.datetime object, the delta is then added, and the result is dropped back down to a datetime.time object.
    
for i in data[data.air_time.isnull()].index.tolist():
    origin=data.iloc[i,3]
    dest=data.iloc[i,6]
    t1 = dt.timedelta(hours=origin.hour,minutes=origin.minute)
    t2 = dt.timedelta(hours=dest.hour,minutes=dest.minute)
    data.iloc[i,14]=(t2-t1).seconds/60
data[data.isnull().any(axis=1)]

data['dep_timestamp']=data[['year','month','day','hour','minute']].apply(lambda x: dt.datetime(*x),axis=1)


data=data.drop(columns=['time_hour'])
flights = data.copy()
#data_dropna[data_dropna['dest']=='SEA'].groupby('dest').size() "Alternative method"
flights[flights['dest']=='SEA'].dest.value_counts()
flights[flights['dest']=='SEA'].carrier.nunique()
flights[flights['dest']=='SEA'].tailnum.nunique()
flights[flights['dest']=='SEA'].arr_delay.mean()
flight_per_airport = flights[flights['dest']=='SEA'].groupby('origin').size().reset_index(name = 'flight_count')
flight_per_airport.loc[2] = ['LGA',0]
flight_per_airport['Proportion'] = (flight_per_airport.flight_count/flight_per_airport.flight_count.sum())*100
flight_per_airport
flights['dep_timestamp'] = pd.to_datetime(flights['dep_timestamp'], errors='coerce') #if the conversion fails for any particular string then those rows are set to NaT.
avg_dep_delay = flights.groupby(flights.dep_timestamp.dt.date)['dep_delay'].mean().sort_values(ascending=False).head(1)
avg_dep_delay
avg_arr_delay = flights.groupby(flights.dep_timestamp.dt.date)['arr_delay'].mean().sort_values(ascending=False).head(1)
avg_arr_delay
avg_dep_delay = flights.groupby(flights.dep_timestamp.dt.date)['dep_delay'].mean().sort_values(ascending=False).head(1)
printmd("*Worst day based on highest average departure delay is : *")
print(avg_dep_delay)
print("\n")

highest_delay = flights.iloc[flights.dep_delay.sort_values(ascending=False).head(2).index[0]][['dep_timestamp','dep_delay']]
printmd("*Worst day based on highest minute departure delay for single flight on a day is : *")
print(highest_delay)
print("\n")


most_delayed=flights[flights.dep_delay>0].groupby(flights.dep_timestamp.dt.date)['dep_delay'].count().sort_values(ascending=False).head(1)
printmd("*Worst day based on most delayed flights on single day is : *")
print(most_delayed)
print("\n")


most_delayed_count=flights[flights.dep_delay>0].groupby(flights.dep_timestamp.dt.date)['dep_delay'].count()
most_delayed_sum=flights.groupby(flights.dep_timestamp.dt.date)['dep_delay'].count()
most_delay_percent = ((most_delayed_count/most_delayed_sum)*100).sort_values(ascending=False).head(1)
printmd("*Worst day based on highest percentage of departure delay on a given day is : *")
print(most_delay_percent)

plt.plot(flights.groupby('month')['dep_delay'].mean())
plt.xlabel('Months')
plt.ylabel('Average_dep_delay')
plt.xlim(1,12)
plt.show()
printmd('*Departure delays are more prominent during jun,july & Dec*')
plt.plot(flights.groupby('hour')['dep_delay'].mean())
plt.xlabel('Hour')
plt.ylabel('Average_dep_delay')
plt.xlim(0,23)
plt.show()
printmd('***Departure delays are more prominent during the evening & night*')
print(flights[flights['arr_delay']>0].groupby(['carrier']).arr_delay.sum().sort_values(ascending=False).head(1))

print(flights[flights['dep_delay']>0].groupby(['carrier']).dep_delay.sum().sort_values(ascending=False).head(1))

printmd("*Above results shows that EV airlines contributed most to the sum total minutes of arrival delay\
        as well as departure delay.*")
print(flights[flights['dep_delay']>0].groupby(['origin','dest']).dep_delay.sum().sort_values(ascending=False).head(1))
print(flights[flights['arr_delay']>0].groupby(['origin','dest']).arr_delay.sum().sort_values(ascending=False).head(1))

printmd("*Above results shows that LA Guardia to ATLANTA route has most amount of delay in minutes generally\
            both in case of arrival as well as departure delay*")
flights['speed']=flights.distance/(flights.air_time/60)
flights[flights.speed==flights.speed.max()]
plt.plot(flights.groupby('carrier')['speed'].mean().sort_values())
plt.ylabel('Miles_per_hour')
plt.xlabel('Carrier')
plt.show()
printmd("*HA carrier has the highest average flight speed*")
flight_group = flights.groupby(['carrier','flight','dest']).size().reset_index(name='counter')
flight_group[flight_group.counter == max(flight_group.counter)]
fly_SEA=flights[flights.dest == 'SEA']
Y1=fly_SEA.groupby('carrier')['dep_delay'].mean()
Y2=fly_SEA.groupby('carrier')['arr_delay'].mean()
X=sorted(fly_SEA.carrier.unique())
plt.subplot(1,2,1)
plt.xlabel('Avg_dep_delay')
plt.bar(X,Y1)
plt.ylim(-15,20)
plt.subplot(1,2,2)
plt.xlabel('Avg_arr_delay')
plt.ylim(-15,20)
plt.bar(X,Y2)
plt.show()
printmd("*UA airlines has a very high average departure delay while AS airlines has a negative average arrival delay means the flights generally arrive earlier than scheduled arrival time.*")
delay = flights.groupby('origin').dep_delay.sum().sort_values()
delay_sum = flights.groupby('origin').dep_delay.sum().sum()
delay_percent = (delay/delay_sum)*100
print(delay_percent)
delay_percent.plot(kind='bar')
plt.ylabel('Departure delay percentage')
plt.show()
printmd("*LGA airport is best with the least departure delay percentage*")
print(flights.groupby(['dest']).flight.count().sort_values(ascending=False).head(1))
plt.figure(figsize=(20,10))
var = flights.groupby(['dest']).flight.count().sort_values()
var.plot(kind='bar')
plt.xlabel('Destination')
plt.ylabel('Flight Count')
plt.show()
printmd("*Chicago O'Hare International Airport is overutilized and recieves highest number of flights,\
        while Blue Grass Airport is underutilized and recieves the least number of flights*")