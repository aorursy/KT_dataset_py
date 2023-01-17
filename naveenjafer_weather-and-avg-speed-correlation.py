import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import os

from math import sin, cos, sqrt, atan2, radians

import datetime as dt

print(os.listdir("../input"))

#from sklearn import model_selection, preprocessing

import xgboost as xgb

color = sns.color_palette()



pd.options.mode.chained_assignment = None  # default='warn'

pd.set_option('display.max_columns', 500)



train_df = pd.read_csv("../input/nyc-taxi-trip-duration/train.csv")

weather = pd.read_csv("../input/weather-data-in-new-york-city-2016/weather_data_nyc_centralpark_2016.csv")



R = 6373.0



print(train_df.shape)

print("Loaded")

print(weather.shape)

#print(weather.iloc[[1]].id)

weatherMatrix = weather.as_matrix()

#print(weatherMatrix[0])

date_dict = {}

count = 0

for date in weather.date:

    precip = weather.iloc[count]['precipitation']

    if str(weather.iloc[count]['precipitation']) == "T":

        precip = 0

    snow_fall = weather.iloc[count]['snow fall']

    if str(weather.iloc[count]['snow fall']) == "T":

        snow_fall = 0

    snow_depth = weather.iloc[count]['snow depth']

    if str(weather.iloc[count]['snow depth']) == "T":

        snow_depth = 0

    date_dict[str(date)] = {'avg_temp':weather.iloc[count]['average temperature'] , 'prec': float(precip) ,  'snow_fall': float(snow_fall) , 'snow_depth':float(snow_depth)}

    count = count + 1





count = 0

for pickup_datetime in train_df.pickup_datetime:

    dateString = str(pickup_datetime)

    dateString = dateString[2:10]

    dateString = dateString[6:] + "-" + dateString[3:5] + "-" + dateString[0:2]

    #print(dateString)

    if not 'trip' in date_dict[dateString]:

        date_dict[dateString]['trip'] = []

        # Create a array

    dataObject = {}

    keysToRead = ['pickup_longitude','pickup_latitude','dropoff_longitude','dropoff_latitude','trip_duration']

    for keys in keysToRead:

        #print(float(train_df.iloc[count][keys]))

        dataObject[keys]  = float(train_df.iloc[count][keys])

    lat1 = radians(abs(dataObject['pickup_latitude']))

    lon1 = radians(abs(dataObject['pickup_longitude']))

    lat2 = radians(abs(dataObject['dropoff_latitude']))

    lon2 = radians(abs(dataObject['dropoff_longitude']))



    dlon = lon2 - lon1

    dlat = lat2 - lat1



    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2

    c = 2 * atan2(sqrt(a), sqrt(1 - a))



    distance = R * c

    dataObject['distance'] = distance

    dataObject['speed'] = (distance * 60 * 60)/dataObject['trip_duration']

    date_dict[dateString]['trip'].append(dataObject)

    count = count + 1

    if(count % 100000 == 0):

        # Remove this break and run the results.

        break

        print(str(count) + ' finished processing')

#print("completed")

#print len(date_dict['21-10-16']['trip'])

#print(date_dict)



count = 0

dates = []

rainValid = []

datesValid = []

snowFallValid = []

snowDepthValid = []

tripsByDate = []

speedList = []

avgSpeedList = []

plt.figure(figsize=(12,6))



#fig, ax = plt.subplots()

for date in weather.date:

    if 'trip' in date_dict[str(date)]:

        # calculate avg speed.

        speedSum = 0

        modifyDate = str(date)

        modifyDate = modifyDate[0:6] + "20" + modifyDate[6:]

        dateObject = dt.datetime.strptime(modifyDate,'%d-%m-%Y').date()

        for tripDetail in date_dict[str(date)]['trip']:

            dates.append(dateObject)

            speedList.append(tripDetail['speed'])

            speedSum = speedSum + tripDetail['speed']

        date_dict[str(date)]['avgSpeed'] = speedSum / len(date_dict[str(date)]['trip'])

        datesValid.append(dateObject)

        rainValid.append(date_dict[str(date)]['prec'])

        snowFallValid.append(date_dict[str(date)]['snow_fall'])

        snowDepthValid.append(date_dict[str(date)]['snow_depth'])

        avgSpeedList.append(date_dict[str(date)]['avgSpeed'])

        tripsByDate.append(len(date_dict[str(date)]['trip']))

    else:

        break

    count = count + 1



plt.suptitle('Relation between number of trips and Snow Depth', fontsize=14, fontweight='bold')

plt.subplot(211)

plt.plot(datesValid, tripsByDate)

plt.ylabel('Number of trips',fontsize = 12)

plt.grid()



#ax2 = fig1

plt.subplot(212)

plt.plot(datesValid, snowDepthValid)

plt.ylabel('Snow Depth',fontsize = 12)

plt.grid()

plt.show()





plt.suptitle('Scatter plot of all speeds on given day to understand outliers', fontsize=14, fontweight='bold')

plt.scatter(dates,speedList,s=2,c='r')

plt.ylabel('All speeds on given day', fontsize = 12)

plt.grid()

plt.show()



plt.suptitle('Relation between average Speed and Snow Depth', fontsize=14, fontweight='bold')

#ax3 = fig1

plt.subplot(211)

plt.plot(datesValid,avgSpeedList)

plt.ylabel('Average speed', fontsize = 12)

plt.grid()



plt.subplot(212)

plt.plot(datesValid, snowDepthValid)

plt.ylabel('Snow Depth',fontsize = 12)

plt.grid()

plt.show()



#ax4 = fig1

plt.suptitle('Rain vs Average Speed Scatter Plot', fontsize=14, fontweight='bold')

plt.scatter(avgSpeedList,rainValid,s=5,c='r')

plt.ylabel('Rain', fontsize = 12)

plt.xlabel('Average speed', fontsize = 12)

plt.grid()

plt.show()



#ax5 = fig1

plt.suptitle('Snow fall vs Average Speed Scatter Plot', fontsize=14, fontweight='bold')

plt.scatter(avgSpeedList,snowFallValid,s=5,c='r')

plt.ylabel('Snow fall', fontsize = 12)

plt.xlabel('Average speed', fontsize = 12)

plt.grid()

plt.show()



#ax6 = fig1

plt.suptitle('Snow Depth vs Average Speed Scatter Plot', fontsize=14, fontweight='bold')

plt.scatter(avgSpeedList,snowDepthValid,s=5,c='r')

plt.ylabel('Snow depth', fontsize = 12)

plt.xlabel('Average speed', fontsize = 12)

plt.grid()

plt.show()



print("Completed Execution")

quit()