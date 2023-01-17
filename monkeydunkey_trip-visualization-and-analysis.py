# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

from matplotlib import rcParams  

%matplotlib inline



# Setting for figures

pd.options.display.mpl_style = 'default' #Better Styling  

new_style = {'grid': False} #Remove grid  

matplotlib.rc('axes', **new_style)  



rcParams['figure.figsize'] = (17.5, 10) #Size of figure  

rcParams['figure.dpi'] = 250

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
tripData = pd.read_csv('../input/train.csv')

tripData.head()
#Checking for nulls in data

tripData.isnull().any()
#So we there are no nulls that is good for processing

#Lets check the min max longitude and lattitude

print ('Longitude min max', min(tripData.pickup_longitude.min(), tripData.dropoff_longitude.min()), max(tripData.pickup_longitude.max(), tripData.dropoff_longitude.max()))

print ('Latitude min max', min(tripData.pickup_latitude.min(), tripData.dropoff_latitude.min()), max(tripData.pickup_latitude.max(), tripData.dropoff_latitude.max()))
# Defining the box to work with

min_long = -74.25

max_long = -73.7

min_lat = 40.6

max_lat = 40.9



def filter_long(longi):

    return longi >= min_long and longi <= max_long



def filter_lat(lat):

    return lat >= min_lat and lat <= max_lat



tripData = tripData[(tripData['pickup_longitude'].apply(filter_long)) & (tripData['dropoff_longitude'].apply(filter_long))]

tripData = tripData[(tripData['pickup_latitude'].apply(filter_lat)) & (tripData['dropoff_latitude'].apply(filter_lat))]
f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)





P_pickups = tripData.plot(kind='scatter', x='pickup_longitude', y='pickup_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, subplots=True, ax=ax1)

ax1.set_title("Aggregate Pickups")

ax1.set_axis_bgcolor('black') #Background Color



P_dropoff = tripData.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, subplots=True, ax=ax2)

ax2.set_title("Aggregate DropOffs")

ax2.set_axis_bgcolor('black') #Background Color
#Defining Boxes around the airports - JFK and LaGuardia

#First lets check the trips that start from JKF

# JFK Loc: 40.6413, -73.7781

JFK = {

    "minLat": 40.62,

    "maxLat": 40.68,

    "minLong": -73.81,

    "maxLong": -75.75

}



JFKData = tripData[(tripData['pickup_longitude'].apply(lambda x: (x >=JFK["minLong"]) & (x <= JFK["maxLong"])))]

JFKData = tripData[(tripData['pickup_latitude'].apply(lambda x: (x >=JFK["minLat"]) & (x <= JFK["maxLat"])))]

P_JFK = JFKData.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, title="Drop offs from JFK")

P_JFK.set_axis_bgcolor('black') #Background Color
#Defining Boxes around the airports - JFK and LaGuardia

# La Guardia Loc: 40.7769, -73.8740

LaGuardia = {

    "minLat": 40.76,

    "maxLat": 40.78,

    "minLong": -73.895,

    "maxLong": -73.855

}



LaGuardiaData = tripData[(tripData['pickup_longitude'].apply(lambda x: (x >=LaGuardia["minLong"]) & (x <= LaGuardia["maxLong"])))]

LaGuardiaData = tripData[(tripData['pickup_latitude'].apply(lambda x: (x >=LaGuardia["minLat"]) & (x <= LaGuardia["maxLat"])))]

P_LaGuardia = LaGuardiaData.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, title="Drop offs from La Guardia")

P_LaGuardia.set_axis_bgcolor('black') #Background Color
#There does not seems much evidence to support my assumption that the dispersed drop offs are due to people taking cabs

#from airports to their homes 

#Interesting La Guardia has a lot more drop offs with a higher concentration in Manhattan where as for JFK the 

#drop offs are uniform in nature. 

#Lets check the number to understand the traffic from each airport

print ('Pickups near La Guardia', LaGuardiaData.shape[0])

print ('Pickups near JFK', JFKData.shape[0])
# Morning Rush Hour 7 AM - 12 PM

tripData['pickup_hour'] = pd.to_datetime(tripData['pickup_datetime']).dt.hour

morningData = tripData[(tripData.pickup_hour >= 7) & (tripData.pickup_hour < 12)]



P_morning = morningData.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, title="Drop Offs during Morning")

P_morning.set_axis_bgcolor('black') #Background Color
# AfterNoon 12 PM - 5 PM

afterNoon = tripData[(tripData.pickup_hour >= 12) & (tripData.pickup_hour < 17)]



P_afterNoon = afterNoon.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, title="Drop Offs during AfterNoon")

P_afterNoon.set_axis_bgcolor('black') #Background Color
# Evening 5 PM - 9 PM

eveningData = tripData[(tripData.pickup_hour >= 17) & (tripData.pickup_hour < 21)]



P_eveningData = eveningData.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, title="Drop Offs during Evening")

P_eveningData.set_axis_bgcolor('black') #Background Color
nightData = tripData[(tripData.pickup_hour >= 21) | (tripData.pickup_hour < 7)]



P_nightData = nightData.plot(kind='scatter', x='dropoff_longitude', y='dropoff_latitude',

                color='white', xlim=(min_long,max_long), ylim=(min_lat, max_lat),

                s=.02, alpha=.6, title="Drop Offs during Night")

P_nightData.set_axis_bgcolor('black') #Background Color
#Lets also look at the number of trips during the 4 divisions that we defined above

print('# of Trips in Morning', morningData.shape[0])

print('# of Trips in Afternoon', afterNoon.shape[0])

print('# of Trips in Evening', eveningData.shape[0])

print('# of Trips in Night', nightData.shape[0])