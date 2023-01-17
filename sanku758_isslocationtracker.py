# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.

# !pip install libgeos-3.5.0

# !pip install libgeos-dev

!pip install https://github.com/matplotlib/basemap/archive/master.zip

!pip install pytrends
from pytrends.request import TrendReq

import pandas as pd

import matplotlib.pyplot as plt

from IPython.display import Image

import requests as r

import matplotlib

from mpl_toolkits.basemap import Basemap

import time

from datetime import datetime
# See who are the people on ISS at current time

people_on_iss = r.get(url='http://api.open-notify.org/astros.json')

people_on_iss.json()
# See the current location of ISS

current_loc = r.get(url='http://api.open-notify.org/iss-now.json')

current_location = current_loc.json()

current_location
# Parsing the json to get our data

timestamp = current_location['timestamp']

longitude = current_location['iss_position']['longitude']

latitude = current_location['iss_position']['latitude']

timestamp, longitude, latitude
type(timestamp), type(longitude), type(latitude)
longitude = float(longitude)

latitude = float(latitude)

type(timestamp), type(longitude), type(latitude)
plt.figure(figsize=(16,8))



m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha=0.3)

m.drawcoastlines(linewidth=0.1, color="white")

# This will draw a world map
plt.figure(figsize=(16,8))



m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha=0.3)

m.drawcoastlines(linewidth=0.1, color="white")



#using scatter plot to plot a point of our ISS location on map

m.scatter(longitude, latitude, s=500, alpha=0.4,color='blue')

plt.title('International Space Station Location' , fontsize=30) 
def get_current_loc():

    current_loc = r.get(url='http://api.open-notify.org/iss-now.json')

    current_location = current_loc.json()

    timestamp = current_location['timestamp']

    longitude = float(current_location['iss_position']['longitude'])

    latitude = float(current_location['iss_position']['latitude'])

    return timestamp, longitude, latitude



timp, long, lat = get_current_loc()

timp, long, lat
import time



def get_current_loc():

    current_loc = r.get(url='http://api.open-notify.org/iss-now.json')

    current_location = current_loc.json()

    timestamp = current_location['timestamp']

    longitude = float(current_location['iss_position']['longitude'])

    latitude = float(current_location['iss_position']['latitude'])

    return timestamp, longitude, latitude



space_station_data = []

while True:

    timp, long, lat = get_current_loc()

    space_station_data.append([timp, long, lat]) # just appending values to the data list

    

    if len(space_station_data) > 20:

        break

    print(space_station_data[-1])

    # finally we want to take data after every 60 seconds thats why

    time.sleep(60)

        



# lets create a pandas dataframe using this data which we will use later

df = pd.DataFrame(space_station_data, columns=['Timestamp', 'Longitude', 'Latitude'])   



# Save the dataframe to a csv file

df.to_csv('ISS_Location_Data.csv', index ='None')

print(space_station_data[:5])
iss_data = pd.read_csv('ISS_Location_Data.csv') # Reading our data to a data frame

iss_data.shape # Lets check our the shape quickly
# Converting timestamp to an actual date format

date = [datetime.fromtimestamp(dt) for dt in iss_data['Timestamp']]

iss_data['Date'] = date



#Indexing our collumn

iss_data['Index'] = range(0, len(iss_data))



#Lets check our data

iss_data.head()
plt.figure(figsize=(16,8))



m=Basemap(llcrnrlon=-180, llcrnrlat=-65,urcrnrlon=180,urcrnrlat=80)

m.drawmapboundary(fill_color='#A6CAE0', linewidth=0)

m.fillcontinents(color='grey', alpha=0.3)

m.drawcoastlines(linewidth=0.1, color="white")



#using scatter plot to plot a point of our ISS location on map

m.scatter(iss_data.Longitude, iss_data.Latitude, s=iss_data.Index, alpha=0.4,color='blue')

plt.title('International Space Station Location' , fontsize=30) 
plt.plot(iss_data.Longitude,iss_data.Latitude)

plt.grid()

plt.title('Longitude vs Latitude')

plt.show()
from scipy import stats 

slope, intercept, r_value, p_value, std_error = stats.linregress(x = iss_data.Longitude, y = iss_data.Latitude)



print('intercept: ',intercept)

print('slope:', slope)
predicted_latitude = np.ceil(intercept + slope * -40)

print("Next predicted latitude is: ", predicted_latitude)
fig, ax = plt.subplots(figsize=(10, 6))



plt.plot(iss_data.Longitude, iss_data.Latitude)

plt.suptitle('Prediction of future latitude of ISS')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.grid()



plt.scatter([-40], [predicted_latitude], color='r')
x = iss_data['Longitude']

y = iss_data['Latitude']

 

polyreg = np.poly1d(np.polyfit(x, y, 3)) #This will basically create a polynomial model
predicted_latitude = polyreg(-40)

print('Next latitude: ',predicted_latitude)
fig, ax = plt.subplots(figsize=(10, 6))



plt.plot(iss_data.Longitude, iss_data.Latitude)

plt.suptitle('Prediction of future latitude of ISS')

plt.xlabel('Longitude')

plt.ylabel('Latitude')

plt.grid()



plt.scatter([-40], [predicted_latitude], color='r')