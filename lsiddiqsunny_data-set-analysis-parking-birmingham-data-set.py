import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from scipy.stats import pearsonr

%matplotlib inline
parking_data = pd.read_csv('/kaggle/input/dataset.csv')

parking_data.describe()
print('Before removing inconsistence data:',parking_data.shape)

parking_data.dropna(inplace = True)

parking_data.drop_duplicates(keep='first',inplace=True) 

parking_data = parking_data[parking_data['Occupancy']>=0 ]

parking_data = parking_data[parking_data['Capacity']>=0 ]

false_data = parking_data[parking_data['Occupancy']> parking_data['Capacity']]

parking_data = pd.concat([parking_data, false_data]).drop_duplicates(keep=False)

print('After removing inconsistence data:',parking_data.shape)

parking_data.describe()
parking_data['OccupancyRate'] = (100.0*parking_data['Occupancy'])/parking_data['Capacity']

dateTime = parking_data['LastUpdated'].str.split(" ", n = 1, expand = True) 

date = dateTime[0]

time = dateTime[1]

parking_data['Date'] = date

parking_data['Time'] = time

day_name = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

parking_data['DayOfWeek'] = pd.to_datetime(parking_data['Date']).dt.dayofweek.apply(lambda x: day_name[x])

parking_data.info()
plt.plot([],[])

park_name = parking_data['SystemCodeNumber'].unique()

#print(park_name)

for i in range(len(park_name)):

    s = park_name[i]

    rate = parking_data[parking_data['SystemCodeNumber'] == s]['OccupancyRate']

    time=pd.to_datetime(parking_data[parking_data['SystemCodeNumber'] == s]['Time'],format='%H:%M:%S')

    plt.scatter(time,rate,label=s)

plt.gcf().autofmt_xdate()

myFmt = mdates.DateFormatter('%H:%M')

plt.gca().xaxis.set_major_formatter(myFmt)    

plt.xlabel('Time')

plt.ylabel('Occupancy Rate')

plt.show()

plt.close()
xData = parking_data.groupby('SystemCodeNumber')['OccupancyRate'].mean()

key_list = list(xData.keys()) 

val_list = []

for x in key_list:

    val_list.append(xData[x])

df = pd.DataFrame(list(zip(key_list, val_list)), 

               columns =['Park ID', 'Mean Occupancy Rate']) 

ax = sns.barplot(y='Park ID',x='Mean Occupancy Rate',data=df,orient="h")

ax.set(ylabel="Car Park ID", xlabel = "Mean Occupancy Rate")

ax.tick_params(axis='y', labelsize=7)

for i in range(1):

    s = park_name[i]

    rate = parking_data[parking_data['SystemCodeNumber'] == s]['OccupancyRate']

    time=pd.to_datetime(parking_data[parking_data['SystemCodeNumber'] == s]['Time'],format='%H:%M:%S')

    plt.scatter(time,rate,label=s)

plt.gcf().autofmt_xdate()

myFmt = mdates.DateFormatter('%H:%M')

plt.gca().xaxis.set_major_formatter(myFmt)    

plt.xlabel('Time')

plt.ylabel('Occupancy Rate')

plt.show()

plt.close()
park_name = ['BHMEURBRD01']

for i in range(len(park_name)):

    s = park_name[i]

    rate = parking_data[parking_data['SystemCodeNumber'] == s]['OccupancyRate']

    time=pd.to_datetime(parking_data[parking_data['SystemCodeNumber'] == s]['Date'])

    plt.plot(time,rate)

    plt.gcf().autofmt_xdate()

plt.xlabel('Date')

plt.ylabel('Occupancy Rate')

plt.show()

plt.close()
plt.plot([],[])



rate = parking_data[parking_data['SystemCodeNumber'] == 'Shopping']

rate = rate[rate['Date'] == '2016-10-06']['OccupancyRate']

time = parking_data[parking_data['SystemCodeNumber'] == 'Shopping'] 

time=pd.to_datetime(time[time['Date'] == '2016-10-06']['Time'],format='%H:%M:%S')

plt.plot(time,rate,label='2016-10-06')



rate = parking_data[parking_data['SystemCodeNumber'] == 'Shopping']

rate = rate[rate['Date'] == '2016-10-09']['OccupancyRate']

time = parking_data[parking_data['SystemCodeNumber'] == 'Shopping'] 

time=pd.to_datetime(time[time['Date'] == '2016-10-09']['Time'],format='%H:%M:%S')

plt.plot(time,rate,label='2016-10-09')







plt.gcf().autofmt_xdate()

myFmt = mdates.DateFormatter('%H:%M')

plt.gca().xaxis.set_major_formatter(myFmt)

plt.xlabel('Time')

plt.ylabel('Occupancy Rate')

plt.legend()

plt.show()

plt.close()
ax = sns.catplot(x='DayOfWeek',kind='count',data=parking_data,orient="h")

ax.fig.autofmt_xdate()

ax.set(xlabel="Week Days", ylabel = "Count")
ax = sns.catplot(x = "DayOfWeek",y="OccupancyRate",kind='box',data=parking_data)

ax.set(xlabel="Week Days", ylabel = "Occupancy Rate")
heatmap_data = pd.pivot_table(parking_data, values='OccupancyRate', 

                     index=['SystemCodeNumber'], 

                     columns='Date')

ax = sns.heatmap(heatmap_data , cmap="BuGn")

ax.set(ylabel="Car Park ID", xlabel = "Date")
test_data = parking_data[(pd.to_datetime(parking_data['Date']) >= pd.to_datetime('2016-12-13'))]

train_data = pd.concat([parking_data, test_data]).drop_duplicates(keep=False)

print('Train data size:',train_data.shape)

print('Test data size:',test_data.shape)

train_data.to_csv('train.csv',index=False)

test_data.to_csv('test.csv',index=False)
