# importing the librarys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.parser import parse # Helps to format strins into date

%matplotlib inline
# importing each dataset

trip = pd.read_csv('../input/trip.csv', error_bad_lines=False)


station = pd.read_csv('../input/station.csv',error_bad_lines=False)


weather = pd.read_csv('../input/weather.csv',error_bad_lines=False)
trip.head(10)
trip.info()
station.head()
station.info()
weather.head()
weather.info()
trip.columns
# lets drop the id columns that that are unnecessary

trip.drop(['trip_id','bikeid'], axis = 1,   inplace = True) # This command drop off the columns we pass as argument, the axis=1 condition,
#makes the drop on columns, the inplace = True, save the alteration on the dataset
trip.info()
# Tranforming the starttime column

data = trip['starttime']
dta = list(trip['starttime']) # transforms each element of the startime column into a string
dta = pd.to_datetime(dta)  # In the format of string, each element of the list is transformed into date by the pandas
trip['starttime'] = dta # Saving column changes
# Checking if everything went right
trip.info()
trip.starttime
trip.head()
# in fact we just need the starttime column, since we already have the variable duration of the trip

trip.drop(columns='stoptime', axis = 1, inplace=True)
trip.info()
trip.isnull().sum()
trip.birthyear.describe()
# Filling in the missing values with values between 1969 and 1989 (which is the range in which most of the data is).
trip.birthyear.fillna(value = np.random.randint(1969,1989), inplace=True)
    
trip.info()
trip.birthyear.describe()
year = trip.starttime

def age(year):
    '''This function extracts the year from each element of the starttime column' '''
    age = []
    for i in year.index:  # get each element in the index of the variable 'year'
        a = str(year[i])  # 'i' represents each element of the index of the variable 'year', so each time 'for' identifies a
        # number in the index it plays within the variable 'a' that selects an item from the variable 'year' 
        b = a.split('-')[0] # variable 'b', stores the result of the .split () method applied on variable 'a', in
        # Then I extract the first element of the result from .split (), which is the year
        c = pd.to_numeric(b)  # converts the string year, to number
        

        age.append(c.astype(int)) # stores 'c' in the 'age' list, created at the beginning of the function
    return age
# usando a função e armazenando o resultado em uma variável
aged = age(year)
trip['age'] = aged - trip.birthyear
trip['age'] = trip['age'].astype(int)
trip.columns
trip.head()
# Populating missing values from the gender column
trip.gender.value_counts()
trip.gender.isnull().sum()
# Using the fillna method with the 'ffill' parameter to populate the null values ​​with the next valid observation of the dataset
gender = trip.gender.fillna(method='ffill')
trip.gender = gender
trip.gender.isnull().sum()
trip.head()
station.head()
# creating the 'from_station_id' and 'to_station_id' columns in the dataset station

station['from_station_id'] = station.station_id
station['to_station_id'] = station.station_id
station.head()

# creating another dataset, only with the 'from_station_id' column and the location data
from_station = station[['lat', 'long','from_station_id']]
from_station.head()
# Including the latitude and longitude of the start stations in a new dataset: trip2

trip2 = pd.merge(trip,from_station, on='from_station_id')
trip2.info()
# identifying the new columns as the data of the place of departure
trip2.columns = ['starttime', 'tripduration', 'from_station_name',
       'to_station_name', 'from_station_id', 'to_station_id', 'usertype',
       'gender', 'birthyear', 'idade', 'from_lat', 'from_long']
trip2.columns
# creating another dataset, only with the column 'to_station_id'
to_station = station[['lat', 'long','to_station_id']]
to_station.head()
# Including the latitude and longitude of the start stations in a new dataset: trip3

trip3 = pd.merge(trip2,to_station, on='to_station_id')
trip3.columns
# identifying the columns of the arrival data
trip3.columns = ['starttime', 'tripduration', 'from_station_name',
       'to_station_name', 'from_station_id', 'to_station_id', 'usertype',
       'gender', 'birthyear', 'idade', 'from_lat', 'from_long', 'to_lat', 'to_long']
trip3.head()
trip3.info()
# Folium is the library that allows plotting with maps, very simple to use

import folium
station.columns
mapa = folium.Map(location=[ 47.608013,  -122.335167], zoom_start=12) # Determining the seattle map using latitude and longitude data
lat = station['lat'].values # taking the latitude values from the stations of the dataset station
long = station['long'].values # taking the values of longitude of the stations of the dataset station

for la, lo in zip(lat, long): # for each value in lat and long...
    folium.Marker([la, lo]).add_to(mapa) # create a marker and place in the map variable (which in this case is the map of Seattle)
mapa # Show the Map
trip3.from_station_name.value_counts().head(10)

estacoes_mais_pop = pd.DataFrame(trip3.from_station_name.value_counts().head(10)) # Counting the 10 plus creating a new df to be able to pass
# for the folium
station_2 = station[['name','lat', 'long' ]]
station_2.columns = ['from_station_name','lat', 'long']
estacoes_mais_pop = estacoes_mais_pop.reset_index() # resetting the index to adjust the name of the columns
estacoes_mais_pop # note that the column with the station name is named 'index'
estacoes_mais_pop.columns = ['from_station_name','contagem'] # Correcting the problem by simply renaming the columns
estacoes_mais_pop
estacoes_mais_pop = pd.merge(estacoes_mais_pop, station_2, on='from_station_name') # including location data (lat and long) using merge again
estacoes_mais_pop
mapa2 = folium.Map(location=[47.608013,  -122.335167], zoom_start=13) # Same process as above, but we need to create a new Map

lat = estacoes_mais_pop['lat'] 
long = estacoes_mais_pop['long'] 

# This time I wrote line by line because I wanted to include the name of the station on the map. I could not find a more practical way to do it,
# for a while...

folium.Marker([47.614315, -122.354093],popup='Pier 69 / Alaskan Way & Clay St').add_to(mapa2)
folium.Marker([47.615330 ,-122.311752],popup='E Pine St & 16th Ave').add_to(mapa2)
folium.Marker([47.618418 ,-122.350964],popup='3rd Ave & Broad St ').add_to(mapa2)
folium.Marker([47.610185 ,-122.339641],popup='2nd Ave & Pine St').add_to(mapa2)
folium.Marker([47.613628 ,-122.337341],popup='Westlake Ave & 6th Ave').add_to(mapa2)
folium.Marker([47.622063 ,-122.321251],popup='E Harrison St & Broadway Ave E ').add_to(mapa2)
folium.Marker([47.615486 ,-122.318245],popup='Cal Anderson Park / 11th Ave & Pine St').add_to(mapa2)
folium.Marker([47.619859 ,-122.330304],popup='REI / Yale Ave N & John St ').add_to(mapa2)
folium.Marker([47.615829 ,-122.348564],popup='2nd Ave & Vine St').add_to(mapa2)
folium.Marker([47.620712 ,-122.312805],popup='15th Ave E & E Thomas St').add_to(mapa2)

mapa2
# Evaluating the weather dataset
weather.head(10)
# Evaluating the dataset trip3, remembering that this dataset contains the location data
trip3.head()
data_str = list(trip3.starttime) # Creating a new date column, with the same date format as the weather dataset
# this will allow you to add the weather data on the trip.
data_str
data_str = [datetime.strftime(x, '%Y-%m-%d') for x in data_str] # Formatting the column using datetime
data_str[:5]
trip3['Date'] = data_str # Adding the column
type(weather.Date)
trip3.head() # Confirming column
trip3.Date.dtypes
weather.Date.dtypes
# using the same method used in the starttime column of the dataset trip this is necessary because the Date columns of trip3 and weather


dt = list(weather['Date']) # transforms each element of the Date column into a string
dt = pd.to_datetime(dt)  # In the string format, each list element is transformed into a date by the pandas

weather['Date'] = dt # Saving the changes
weather.head()
trip3.Date = pd.to_datetime(trip3.Date)
trip4 = pd.merge(weather,trip3, on = 'Date')
trip4.info()
trip4.head()
# We have 110 null values in this colunm
trip4.Mean_Temperature_F.isnull().sum()
trip4.Mean_Temperature_F.describe()
# Let us fill in the missing data with the mean value, plus or minus the standard deviation
trip4.Mean_Temperature_F = trip4.Mean_Temperature_F.fillna(value = np.random.randint(48,68))
trip4.Mean_Temperature_F.describe()
trip4.Mean_Temperature_F.isnull().sum()
trip4.isnull().sum()
trip4.Max_Gust_Speed_MPH.describe()
trip4.drop(columns='Max_Gust_Speed_MPH', axis=1, inplace=True)
trip4.isnull().sum()
trip4.Events.describe()
trip4.Events.value_counts()
events = trip4.Events
events.replace('Rain , Thunderstorm', 'Rain-Thunderstorm', inplace = True)
events.replace('Rain , Snow', 'Rain-Snow', inplace = True)
events.replace('Fog , Rain', 'Rain-Snow', inplace = True)
events.value_counts()
events.fillna(value='No-Event', inplace=True)
events.isnull().sum()
events.value_counts()
trip4.info()
columns_to_drop = ['Max_Temperature_F','Min_TemperatureF', 'Max_Dew_Point_F', 'MeanDew_Point_F', 'Min_Dewpoint_F',
                   'Max_Humidity', 'Min_Humidity','Max_Sea_Level_Pressure_In', 'Min_Sea_Level_Pressure_In', 'Max_Visibility_Miles',
                   'Min_Visibility_Miles', 'Max_Wind_Speed_MPH']                 
# converting the trip duration from seconds to minutes
trip4.tripduration = trip4.tripduration / 60
trip5 = trip4.drop(columns= columns_to_drop, axis=1)
trip5.columns
trip5.head()
trip5.describe()
