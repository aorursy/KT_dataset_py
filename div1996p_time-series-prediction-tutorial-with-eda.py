import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

import folium

import numpy as np
operations = pd.read_csv("../input/world-war-ii/operations.csv")

weather = pd.read_csv("../input/weatherww2/Summary of Weather.csv")

locations = pd.read_csv("../input/weatherww2/Weather Station Locations.csv")
operations.head()
weather.describe()
locations.head()
operations = operations[pd.isna(operations.Country) == False]

operations = operations[pd.isna(operations['Target Longitude']) == False]

operations = operations[pd.isna(operations['Takeoff Longitude']) == False]



drop_list = ['Mission ID','Unit ID','Target ID','Altitude (Hundreds of Feet)','Airborne Aircraft',

             'Attacking Aircraft', 'Bombing Aircraft', 'Aircraft Returned',

             'Aircraft Failed', 'Aircraft Damaged', 'Aircraft Lost',

             'High Explosives', 'High Explosives Type','Mission Type',

             'High Explosives Weight (Pounds)', 'High Explosives Weight (Tons)',

             'Incendiary Devices', 'Incendiary Devices Type',

             'Incendiary Devices Weight (Pounds)',

             'Incendiary Devices Weight (Tons)', 'Fragmentation Devices',

             'Fragmentation Devices Type', 'Fragmentation Devices Weight (Pounds)',

             'Fragmentation Devices Weight (Tons)', 'Total Weight (Pounds)',

             'Total Weight (Tons)', 'Time Over Target', 'Bomb Damage Assessment','Source ID']

operations.drop(drop_list,axis = 1,inplace = True)

operations = operations[ operations.iloc[:,8]!="4248"] # drop this takeoff latitude 

operations = operations[ operations.iloc[:,9]!=1355]  
operations.info()
locations.info()
locations = locations[['WBAN','NAME','STATE/COUNTRY ID','Latitude','Longitude']]

locations.info()
weather.info()
weather = weather[["STA","Date","MeanTemp"]]

weather.info()
weather.head()
operations.head()
#How many country which attacks

counts = operations.Country.value_counts()

print(counts)

plt.figure(figsize = (22,10))

sns.countplot(operations.Country)

plt.show()
#top 10 Aircraft Series

print(operations['Aircraft Series'].value_counts()[0:10])

plt.figure(figsize = (22,10))

sns.countplot(operations['Aircraft Series'])

plt.show()
#top target countries

print(operations['Target Country'].value_counts()[0:10])

plt.figure(figsize=(22,10))

sns.countplot(operations['Target Country'])

plt.xticks(rotation = 90)

plt.show()
map = folium.Map(location=[0,0],zoom_start = 4,tiles = 'Stamen Terrain')



for index,row in operations.iterrows():

    try:

        fg = folium.map.FeatureGroup()

        fg.add_child(folium.CircleMarker(

        [float(row['Takeoff Longitude']),float(row['Takeoff Latitude'])],

        radius = 5,

        color = 'red',

        fill_color = 'red'

        ))



        map.add_child(fg)

        folium.Marker([float(row['Takeoff Longitude']),float(row['Takeoff Latitude'])],

        popup = row["Takeoff Location"]).add_to(map)

    except:

        continue

map
weather.head()
weather_station_id = locations[locations.NAME == 'BINDUKURI']

weather_bin = weather[weather.STA == 32907]

weather_bin['Date'] = pd.to_datetime(weather_bin['Date'])

plt.figure(figsize=(22,10))

plt.plot(weather_bin.Date,weather_bin.MeanTemp)

plt.show()
timeseries = weather_bin[["Date","MeanTemp"]]

timeseries.index = timeseries.Date

ts = timeseries.drop("Date",axis=1)
# adfuller library 

from statsmodels.tsa.stattools import adfuller

def check_adfuller(ts):

    #dickey fullar test

    result = adfuller(ts,autolag='AIC')

    print('Test Statistics:',result[0])

    print('P Value',result[1])

    print('Critical Value',result[4])

#check mean std

def check_mean_std(ts):

    #rolling statistics

    rolmean = pd.rolling_mean(ts,window=6)

    rolstd = pd.rolling_std(ts,window=6)

    plt.figure(figsize=(22,10))

    orgi = plt.plot(ts,color='red',label='Original')

    mean = plt.plot(rolmean,color = 'black',label='Rolling Mean')

    std = plt.plot(rolstd,color='green',label='Rolling STD')

    plt.xlabel("Date")

    plt.ylabel("Mean Temprature")

    plt.title("Rolling Mean & Standard Deviation")

    plt.legend()

    plt.show()

    

check_mean_std(ts)

check_adfuller(ts.MeanTemp)

    
window_size = 6

moving_avg = pd.rolling_mean(ts,window_size)

plt.figure(figsize=(22,10))

plt.plot(ts,color='red',label='Original')

plt.plot(moving_avg,color='black',label="moving avg mean")

plt.title("Mean Temperature of Bindukuri Area")

plt.xlabel("Date")

plt.ylabel("Mean Temperature")

plt.legend()

plt.show()
ts_moving_avg_diff = ts - moving_avg

ts_moving_avg_diff.dropna(inplace=True)

check_mean_std(ts_moving_avg_diff)

check_adfuller(ts_moving_avg_diff.MeanTemp)
ts_diff = ts - ts.shift()

plt.figure(figsize=(22,10))

plt.plot(ts_diff)

plt.title("Differencing method") 

plt.xlabel("Date")

plt.ylabel("Differencing Mean Temperature")

plt.show()
ts_diff.dropna(inplace=True)

check_mean_std(ts_diff)

check_adfuller(ts_diff.MeanTemp)
from statsmodels.tsa.stattools import acf, pacf

lag_acf = acf(ts_diff,nlags=20)

lag_pacf = pacf(ts_diff,nlags=20,method='ols')

# ACF

plt.figure(figsize=(22,10))



plt.subplot(121) 

plt.plot(lag_acf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')

plt.title('Autocorrelation Function')



# PACF

plt.subplot(122)

plt.plot(lag_pacf)

plt.axhline(y=0,linestyle='--',color='gray')

plt.axhline(y=-1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')

plt.axhline(y=1.96/np.sqrt(len(ts_diff)),linestyle='--',color='gray')

plt.title('Partial Autocorrelation Function')

plt.tight_layout()
# ARIMA LİBRARY

from statsmodels.tsa.arima_model import ARIMA

from pandas import datetime

#Model Training

model = ARIMA(ts,order=(1,0,1))

model_fit = model.fit(disp=0)

#Model Testing

intial_index = datetime(1944,6,25)

end_index = datetime(1945,5,31)

forcast = model_fit.predict(start = intial_index,end=end_index)



#visualization



plt.figure(figsize=(22,10))

plt.plot(weather_bin.Date,weather_bin.MeanTemp,label = "Original")

plt.plot(forcast,label="Predicted")

plt.title("Time Series Forecast")

plt.xlabel("Date")

plt.ylabel("Mean Temperature")

plt.legend()

plt.show()