import numpy as np

import matplotlib.pyplot as plt

import datetime

import calendar

import pandas as pd 

import os

import folium

from folium import plugins

from folium.plugins import HeatMap

print(os.listdir("../input"))
data=pd.read_csv('../input/baltimore_crimes.csv')
for crime_time in range(len(data['CrimeTime'])):

    current_crime_time=data['CrimeTime'][crime_time].split(':')

    if len(current_crime_time)==1:

        data['CrimeTime'][crime_time]=current_crime_time[0][0:2]+':'+current_crime_time[0][2:4]+':00'

# data
district_crimes={}

for district in data['District']:

    if district not in district_crimes:

        district_crimes[district]=0

    district_crimes[district]+=1

plt.rcParams["figure.figsize"]=[20,8]

plt.bar(*zip(*district_crimes.items()))

plt.show()
Location=[]

for location in data['Location 1']:

    lt=float(location[1:len(data['Location 1'][0])-1].split(',')[0])

    lg=float(location[1:len(data['Location 1'][0])-1].split(',')[1])

    Location.append((lt,lg))

map_crime = folium.Map(location=[39.2645, -76.6192],

                    zoom_start = 12) 

HeatMap(Location).add_to(map_crime)

map_crime
day_labels = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',  'Friday', 'Saturday','Sunday']

day_crimes=np.zeros(7)

for dates in data['CrimeDate']:

    date=dates.split('/')

    day_crimes[datetime.datetime(int(date[2]),int(date[0]),int(date[1])).weekday()]+=1

plt.rcParams["figure.figsize"]=[20,8]

plt.bar(day_labels,day_crimes)

plt.show()

print("Maximum Crimes occur on "+ calendar.day_name[np.argmax(day_crimes)]  )
crime_time=[]

crime_types={}

for i in range(len(data['CrimeTime'])):

    current_crime_time=data['CrimeTime'][i].split(':')

    current_crime_time[0]=int(current_crime_time[0])

    current_crime_time[1]=int(current_crime_time[1])

    current_crime_time[2]=int(current_crime_time[2])

    if current_crime_time[0] >= 20 and current_crime_time[0]<=23 and current_crime_time[1]<=59 and current_crime_time[1]>=0:

        crime_time.append(i)

for time in crime_time:

    if data['Description'][time] not in crime_types:

        crime_types[data['Description'][time]]=0

    crime_types[data['Description'][time]]+=1

plt.rcParams["figure.figsize"]=[35,10]

plt.bar(*zip(*crime_types.items()))

plt.show()

print("Larcency from auto is the most common crime that occurs between 2000 Hrs and 2359 Hrs. ")
total_crimes={}

for crime in data['Description']:

    if crime not in total_crimes:

        total_crimes[crime]=0

    total_crimes[crime]+=1

plt.rcParams["figure.figsize"]=[35,10]

plt.bar(*zip(*total_crimes.items()))

plt.show()

print("From the plot we can see that Larceny is the most common crime.")
larcency=[]

for i in range(len(data['Description'])):

    if data['Description'][i] == 'LARCENY':

        larcency.append(i)

larceny_time=[]

larceny_location=[]

for i in larcency:

    larceny_time.append(data['CrimeTime'][i])

    larceny_location.append(Location[i])

crime_time=np.zeros(24)

crime

crime_time_labels=[]

for i in range(24):

    crime_time_labels.append(i)

for time in larceny_time:

    time=time.split(':')

    crime_time[int(time[0])]+=1

plt.hist(crime_time_labels,weights=np.array(crime_time),bins=24)

plt.show()
plt.bar(crime_time_labels,np.array(crime_time))

plt.show()

print("From these plots we can see that larceny, which is the most common crime happens mostly around 16:00-17:00")
map_larceny = folium.Map(location=[39.2645, -76.6192],zoom_start=12) 

HeatMap(larceny_location).add_to(map_larceny)

map_larceny