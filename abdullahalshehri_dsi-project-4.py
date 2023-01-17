# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
# reading CSV file as a pandas dataframe for Riyadh city weather

data = pd.read_csv('../input/Riyadh_weather.csv')
# inspecting dataframe head 

data.head()
# inspecting dataframe shape , 13 features

data.shape
# inspecting feature names

data.columns
# checking null values

data.isnull().sum()
# inspect data type , 'date' need change to datetime type

data.info()
data['date'] = pd.to_datetime(data.date)
# inspect data type , now 'date' is a datetime type series

data.info()
# setting date feature as an index to start time series analysis

data.set_index('date', inplace=True)
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')
# ploting everything for initial inspection



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))



ax[0, 0].plot(data.Dew_Point_Avg) #row=0, col=0

ax[1, 0].plot(data.Dew_Point_Max) #row=1, col=0

ax[2, 0].plot(data.Dew_Point_Min) #row=2, col=0

ax[3, 0].plot(data.Humidity_Max) #row=3, col=0

ax[0, 1].plot(data.Humidity_Min) #row=0, col=1

ax[1, 1].plot(data.Pressure_Max) #row=1, col=1

ax[2, 1].plot(data.Pressure_Min) #row=2, col=1

ax[3, 1].plot(data.Temperature_Avg) #row=3, col=1

ax[0, 2].plot(data.Temperature_Max) #row=0, col=2

ax[1, 2].plot(data.Temperature_Min) #row=1, col=2

ax[2, 2].plot(data.Wind_speed_Max) #row=2, col=2

ax[3, 2].plot(data.Wind_speed_Min) #row=3, col=2



ax[0, 0].set_title('Dew_Point_Avg',size=20) #row=0, col=0

ax[1, 0].set_title('Dew_Point_Max',size=20) #row=1, col=0

ax[2, 0].set_title('Dew_Point_Min',size=20) #row=2, col=0

ax[3, 0].set_title('Humidity_Max',size=20) #row=3, col=0

ax[0, 1].set_title('Humidity_Min',size=20) #row=0, col=1

ax[1, 1].set_title('Pressure_Max',size=20) #row=1, col=1

ax[2, 1].set_title('Pressure_Min',size=20) #row=2, col=1

ax[3, 1].set_title('Temperature_Avg',size=20) #row=3, col=1

ax[0, 2].set_title('Temperature_Max',size=20) #row=0, col=2

ax[1, 2].set_title('Temperature_Min',size=20) #row=1, col=2

ax[2, 2].set_title('Wind_speed_Max',size=20) #row=2, col=2

ax[3, 2].set_title('Wind_speed_Min',size=20) #row=3, col=2



plt.show()
# outlier in Dew_Point_Min , it is imposible to have such a value (-117) that 

data.Dew_Point_Min.idxmin() , data.Dew_Point_Min.min()
data.Dew_Point_Min.loc['2009-05-27 00:00:00'] = np.nan
# checking null values to verify deletion of value

data.isnull().sum()
# to mutate NAN value using interpolation

data.Dew_Point_Min.interpolate(method='linear', inplace=True)
# check the value after interpolation

data.Dew_Point_Min.loc['2009-05-27 00:00:00']
# rolling averages

data.Dew_Point_Avg.rolling(30).mean().plot();
# resampling test for monthly to cmpare to rolling average

data.Dew_Point_Avg.resample('M').mean().plot();
# ploting for a weekly resampled data 



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))



ax[0, 0].plot(data.Dew_Point_Avg.resample('W').mean()) #row=0, col=0

ax[1, 0].plot(data.Dew_Point_Max.resample('W').mean()) #row=1, col=0

ax[2, 0].plot(data.Dew_Point_Min.resample('W').mean()) #row=2, col=0

ax[3, 0].plot(data.Humidity_Max.resample('W').mean()) #row=3, col=0

ax[0, 1].plot(data.Humidity_Min.resample('W').mean()) #row=0, col=1

ax[1, 1].plot(data.Pressure_Max.resample('W').mean()) #row=1, col=1

ax[2, 1].plot(data.Pressure_Min.resample('W').mean()) #row=2, col=1

ax[3, 1].plot(data.Temperature_Avg.resample('W').mean()) #row=3, col=1

ax[0, 2].plot(data.Temperature_Max.resample('W').mean()) #row=0, col=2

ax[1, 2].plot(data.Temperature_Min.resample('W').mean()) #row=1, col=2

ax[2, 2].plot(data.Wind_speed_Max.resample('W').mean()) #row=2, col=2

ax[3, 2].plot(data.Wind_speed_Min.resample('W').mean()) #row=3, col=2



ax[0, 0].set_title('Dew_Point_Avg (Resampled Weekly)',size=20) #row=0, col=0

ax[1, 0].set_title('Dew_Point_Max (Resampled Weekly)',size=20) #row=1, col=0

ax[2, 0].set_title('Dew_Point_Min (Resampled Weekly)',size=20) #row=2, col=0

ax[3, 0].set_title('Humidity_Max (Resampled Weekly)',size=20) #row=3, col=0

ax[0, 1].set_title('Humidity_Min (Resampled Weekly)',size=20) #row=0, col=1

ax[1, 1].set_title('Pressure_Max (Resampled Weekly)',size=20) #row=1, col=1

ax[2, 1].set_title('Pressure_Min (Resampled Weekly)',size=20) #row=2, col=1

ax[3, 1].set_title('Temperature_Avg (Resampled Weekly)',size=20) #row=3, col=1

ax[0, 2].set_title('Temperature_Max (Resampled Weekly)',size=20) #row=0, col=2

ax[1, 2].set_title('Temperature_Min (Resampled Weekly)',size=20) #row=1, col=2

ax[2, 2].set_title('Wind_speed_Max (Resampled Weekly)',size=20) #row=2, col=2

ax[3, 2].set_title('Wind_speed_Min (Resampled Weekly)',size=20) #row=3, col=2



plt.show()
# ploting for a monthly resampled data 



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))



ax[0, 0].plot(data.Dew_Point_Avg.resample('M').mean()) #row=0, col=0

ax[1, 0].plot(data.Dew_Point_Max.resample('M').mean()) #row=1, col=0

ax[2, 0].plot(data.Dew_Point_Min.resample('M').mean()) #row=2, col=0

ax[3, 0].plot(data.Humidity_Max.resample('M').mean()) #row=3, col=0

ax[0, 1].plot(data.Humidity_Min.resample('M').mean()) #row=0, col=1

ax[1, 1].plot(data.Pressure_Max.resample('M').mean()) #row=1, col=1

ax[2, 1].plot(data.Pressure_Min.resample('M').mean()) #row=2, col=1

ax[3, 1].plot(data.Temperature_Avg.resample('M').mean()) #row=3, col=1

ax[0, 2].plot(data.Temperature_Max.resample('M').mean()) #row=0, col=2

ax[1, 2].plot(data.Temperature_Min.resample('M').mean()) #row=1, col=2

ax[2, 2].plot(data.Wind_speed_Max.resample('M').mean()) #row=2, col=2

ax[3, 2].plot(data.Wind_speed_Min.resample('M').mean()) #row=3, col=2



ax[0, 0].set_title('Dew_Point_Avg (Resampled Monthly)',size=20) #row=0, col=0

ax[1, 0].set_title('Dew_Point_Max (Resampled Monthly)',size=20) #row=1, col=0

ax[2, 0].set_title('Dew_Point_Min (Resampled Monthly)',size=20) #row=2, col=0

ax[3, 0].set_title('Humidity_Max (Resampled Monthly)',size=20) #row=3, col=0

ax[0, 1].set_title('Humidity_Min (Resampled Monthly)',size=20) #row=0, col=1

ax[1, 1].set_title('Pressure_Max (Resampled Monthly)',size=20) #row=1, col=1

ax[2, 1].set_title('Pressure_Min (Resampled Monthly)',size=20) #row=2, col=1

ax[3, 1].set_title('Temperature_Avg (Resampled Monthly)',size=20) #row=3, col=1

ax[0, 2].set_title('Temperature_Max (Resampled Monthly)',size=20) #row=0, col=2

ax[1, 2].set_title('Temperature_Min (Resampled Monthly)',size=20) #row=1, col=2

ax[2, 2].set_title('Wind_speed_Max (Resampled Monthly)',size=20) #row=2, col=2

ax[3, 2].set_title('Wind_speed_Min (Resampled Monthly)',size=20) #row=3, col=2



plt.show()
# ploting for a quartarly resampled data 



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))



ax[0, 0].plot(data.Dew_Point_Avg.resample('Q').mean()) #row=0, col=0

ax[1, 0].plot(data.Dew_Point_Max.resample('Q').mean()) #row=1, col=0

ax[2, 0].plot(data.Dew_Point_Min.resample('Q').mean()) #row=2, col=0

ax[3, 0].plot(data.Humidity_Max.resample('Q').mean()) #row=3, col=0

ax[0, 1].plot(data.Humidity_Min.resample('Q').mean()) #row=0, col=1

ax[1, 1].plot(data.Pressure_Max.resample('Q').mean()) #row=1, col=1

ax[2, 1].plot(data.Pressure_Min.resample('Q').mean()) #row=2, col=1

ax[3, 1].plot(data.Temperature_Avg.resample('Q').mean()) #row=3, col=1

ax[0, 2].plot(data.Temperature_Max.resample('Q').mean()) #row=0, col=2

ax[1, 2].plot(data.Temperature_Min.resample('Q').mean()) #row=1, col=2

ax[2, 2].plot(data.Wind_speed_Max.resample('Q').mean()) #row=2, col=2

ax[3, 2].plot(data.Wind_speed_Min.resample('Q').mean()) #row=3, col=2



ax[0, 0].set_title('Dew_Point_Avg (Resampled Quartarly)',size=20) #row=0, col=0

ax[1, 0].set_title('Dew_Point_Max (Resampled Quartarly)',size=20) #row=1, col=0

ax[2, 0].set_title('Dew_Point_Min (Resampled Quartarly)',size=20) #row=2, col=0

ax[3, 0].set_title('Humidity_Max (Resampled Quartarly)',size=20) #row=3, col=0

ax[0, 1].set_title('Humidity_Min (Resampled Quartarly)',size=20) #row=0, col=1

ax[1, 1].set_title('Pressure_Max (Resampled Quartarly)',size=20) #row=1, col=1

ax[2, 1].set_title('Pressure_Min (Resampled Quartarly)',size=20) #row=2, col=1

ax[3, 1].set_title('Temperature_Avg (Resampled Quartarly)',size=20) #row=3, col=1

ax[0, 2].set_title('Temperature_Max (Resampled Quartarly)',size=20) #row=0, col=2

ax[1, 2].set_title('Temperature_Min (Resampled Quartarly)',size=20) #row=1, col=2

ax[2, 2].set_title('Wind_speed_Max (Resampled Quartarly)',size=20) #row=2, col=2

ax[3, 2].set_title('Wind_speed_Min (Resampled Quartarly)',size=20) #row=3, col=2



plt.show()
# ploting for a yearly resampled data 



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))



ax[0, 0].plot(data.Dew_Point_Avg.resample('A').mean()) #row=0, col=0

ax[1, 0].plot(data.Dew_Point_Max.resample('A').mean()) #row=1, col=0

ax[2, 0].plot(data.Dew_Point_Min.resample('A').mean()) #row=2, col=0

ax[3, 0].plot(data.Humidity_Max.resample('A').mean()) #row=3, col=0

ax[0, 1].plot(data.Humidity_Min.resample('A').mean()) #row=0, col=1

ax[1, 1].plot(data.Pressure_Max.resample('A').mean()) #row=1, col=1

ax[2, 1].plot(data.Pressure_Min.resample('A').mean()) #row=2, col=1

ax[3, 1].plot(data.Temperature_Avg.resample('A').mean()) #row=3, col=1

ax[0, 2].plot(data.Temperature_Max.resample('A').mean()) #row=0, col=2

ax[1, 2].plot(data.Temperature_Min.resample('A').mean()) #row=1, col=2

ax[2, 2].plot(data.Wind_speed_Max.resample('A').mean()) #row=2, col=2

ax[3, 2].plot(data.Wind_speed_Min.resample('A').mean()) #row=3, col=2



ax[0, 0].set_title('Dew_Point_Avg (Resampled Annually)',size=20) #row=0, col=0

ax[1, 0].set_title('Dew_Point_Max (Resampled Annually)',size=20) #row=1, col=0

ax[2, 0].set_title('Dew_Point_Min (Resampled Annually)',size=20) #row=2, col=0

ax[3, 0].set_title('Humidity_Max (Resampled Annually)',size=20) #row=3, col=0

ax[0, 1].set_title('Humidity_Min (Resampled Annually)',size=20) #row=0, col=1

ax[1, 1].set_title('Pressure_Max (Resampled Annually)',size=20) #row=1, col=1

ax[2, 1].set_title('Pressure_Min (Resampled Annually)',size=20) #row=2, col=1

ax[3, 1].set_title('Temperature_Avg (Resampled Annually)',size=20) #row=3, col=1

ax[0, 2].set_title('Temperature_Max (Resampled Annually)',size=20) #row=0, col=2

ax[1, 2].set_title('Temperature_Min (Resampled Annually)',size=20) #row=1, col=2

ax[2, 2].set_title('Wind_speed_Max (Resampled Annually)',size=20) #row=2, col=2

ax[3, 2].set_title('Wind_speed_Min (Resampled Annually)',size=20) #row=3, col=2



plt.show()
# weekly resampled autocorrelation



from statsmodels.graphics.tsaplots import plot_acf



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Weekly Resampled Autocorrelations', fontsize=20)



plot_acf(data.Dew_Point_Avg.resample('W').mean(),ax=ax[0, 0],lags=480, title='Dew_Point_Avg',) #row=0, col=0

plot_acf(data.Dew_Point_Max.resample('W').mean(), ax=ax[1, 0],lags=480, title='Dew_Point_Max') #row=1, col=0

plot_acf(data.Dew_Point_Min.resample('W').mean(),ax=ax[2, 0],lags=480, title='Dew_Point_Min') #row=2, col=0

plot_acf(data.Humidity_Max.resample('W').mean(),ax=ax[3, 0],lags=480, title='Humidity_Max') #row=3, col=0

plot_acf(data.Humidity_Min.resample('W').mean(),ax=ax[0, 1],lags=480, title='Humidity_Min') #row=0, col=1

plot_acf(data.Pressure_Max.resample('W').mean(),ax=ax[1, 1],lags=480, title='Pressure_Max') #row=1, col=1

plot_acf(data.Pressure_Min.resample('W').mean(),ax=ax[2, 1],lags=480, title='Pressure_Min') #row=2, col=1

plot_acf(data.Temperature_Avg.resample('W').mean(),ax=ax[3, 1],lags=480, title='Temperature_Avg') #row=3, col=1

plot_acf(data.Temperature_Max.resample('W').mean(),ax=ax[0, 2],lags=480, title='Temperature_Max') #row=0, col=2

plot_acf(data.Temperature_Min.resample('W').mean(),ax=ax[1, 2],lags=480, title='Temperature_Min') #row=1, col=2

plot_acf(data.Wind_speed_Max.resample('W').mean(),ax=ax[2, 2],lags=480, title='Wind_speed_Max') #row=2, col=2

plot_acf(data.Wind_speed_Min.resample('W').mean(),ax=ax[3, 2],lags=480, title='Wind_speed_Min') #row=3, col=2



plt.show()
# monthly resampled autocorrelation



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Monthly Resampled Autocorrelations', fontsize=20)



plot_acf(data.Dew_Point_Avg.resample('M').mean(),ax=ax[0, 0],lags=120, title='Dew_Point_Avg',) #row=0, col=0

plot_acf(data.Dew_Point_Max.resample('M').mean(), ax=ax[1, 0],lags=120, title='Dew_Point_Max') #row=1, col=0

plot_acf(data.Dew_Point_Min.resample('M').mean(),ax=ax[2, 0],lags=120, title='Dew_Point_Min') #row=2, col=0

plot_acf(data.Humidity_Max.resample('M').mean(),ax=ax[3, 0],lags=120, title='Humidity_Max') #row=3, col=0

plot_acf(data.Humidity_Min.resample('M').mean(),ax=ax[0, 1],lags=120, title='Humidity_Min') #row=0, col=1

plot_acf(data.Pressure_Max.resample('M').mean(),ax=ax[1, 1],lags=120, title='Pressure_Max') #row=1, col=1

plot_acf(data.Pressure_Min.resample('M').mean(),ax=ax[2, 1],lags=120, title='Pressure_Min') #row=2, col=1

plot_acf(data.Temperature_Avg.resample('M').mean(),ax=ax[3, 1],lags=120, title='Temperature_Avg') #row=3, col=1

plot_acf(data.Temperature_Max.resample('M').mean(),ax=ax[0, 2],lags=120, title='Temperature_Max') #row=0, col=2

plot_acf(data.Temperature_Min.resample('M').mean(),ax=ax[1, 2],lags=120, title='Temperature_Min') #row=1, col=2

plot_acf(data.Wind_speed_Max.resample('M').mean(),ax=ax[2, 2],lags=120, title='Wind_speed_Max') #row=2, col=2

plot_acf(data.Wind_speed_Min.resample('M').mean(),ax=ax[3, 2],lags=120, title='Wind_speed_Min') #row=3, col=2



plt.show()
# Quarterly resampled autocorrelation



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Quarterly Resampled Autocorrelations', fontsize=20)



plot_acf(data.Dew_Point_Avg.resample('Q').mean(),ax=ax[0, 0],lags=40, title='Dew_Point_Avg',) #row=0, col=0

plot_acf(data.Dew_Point_Max.resample('Q').mean(),ax=ax[1, 0],lags=40, title='Dew_Point_Max') #row=1, col=0

plot_acf(data.Dew_Point_Min.resample('Q').mean(),ax=ax[2, 0],lags=40, title='Dew_Point_Min') #row=2, col=0

plot_acf(data.Humidity_Max.resample('Q').mean(),ax=ax[3, 0],lags=40, title='Humidity_Max') #row=3, col=0

plot_acf(data.Humidity_Min.resample('Q').mean(),ax=ax[0, 1],lags=40, title='Humidity_Min') #row=0, col=1

plot_acf(data.Pressure_Max.resample('Q').mean(),ax=ax[1, 1],lags=40, title='Pressure_Max') #row=1, col=1

plot_acf(data.Pressure_Min.resample('Q').mean(),ax=ax[2, 1],lags=40, title='Pressure_Min') #row=2, col=1

plot_acf(data.Temperature_Avg.resample('Q').mean(),ax=ax[3, 1],lags=40, title='Temperature_Avg') #row=3, col=1

plot_acf(data.Temperature_Max.resample('Q').mean(),ax=ax[0, 2],lags=40, title='Temperature_Max') #row=0, col=2

plot_acf(data.Temperature_Min.resample('Q').mean(),ax=ax[1, 2],lags=40, title='Temperature_Min') #row=1, col=2

plot_acf(data.Wind_speed_Max.resample('Q').mean(),ax=ax[2, 2],lags=40, title='Wind_speed_Max') #row=2, col=2

plot_acf(data.Wind_speed_Min.resample('Q').mean(),ax=ax[3, 2],lags=40, title='Wind_speed_Min') #row=3, col=2



plt.show()
# Yearly resampled autocorrelation



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Yearly Resampled Autocorrelations', fontsize=20)



plot_acf(data.Dew_Point_Avg.resample('A').mean(),ax=ax[0, 0],lags=10, title='Dew_Point_Avg',) #row=0, col=0

plot_acf(data.Dew_Point_Max.resample('A').mean(),ax=ax[1, 0],lags=10, title='Dew_Point_Max') #row=1, col=0

plot_acf(data.Dew_Point_Min.resample('A').mean(),ax=ax[2, 0],lags=10, title='Dew_Point_Min') #row=2, col=0

plot_acf(data.Humidity_Max.resample('A').mean(),ax=ax[3, 0],lags=10, title='Humidity_Max') #row=3, col=0

plot_acf(data.Humidity_Min.resample('A').mean(),ax=ax[0, 1],lags=10, title='Humidity_Min') #row=0, col=1

plot_acf(data.Pressure_Max.resample('A').mean(),ax=ax[1, 1],lags=10, title='Pressure_Max') #row=1, col=1

plot_acf(data.Pressure_Min.resample('A').mean(),ax=ax[2, 1],lags=10, title='Pressure_Min') #row=2, col=1

plot_acf(data.Temperature_Avg.resample('A').mean(),ax=ax[3, 1],lags=10, title='Temperature_Avg') #row=3, col=1

plot_acf(data.Temperature_Max.resample('A').mean(),ax=ax[0, 2],lags=10, title='Temperature_Max') #row=0, col=2

plot_acf(data.Temperature_Min.resample('A').mean(),ax=ax[1, 2],lags=10, title='Temperature_Min') #row=1, col=2

plot_acf(data.Wind_speed_Max.resample('A').mean(),ax=ax[2, 2],lags=10, title='Wind_speed_Max') #row=2, col=2

plot_acf(data.Wind_speed_Min.resample('A').mean(),ax=ax[3, 2],lags=10, title='Wind_speed_Min') #row=3, col=2



plt.show()
# Weekly resampled partial autocorrelation



from statsmodels.graphics.tsaplots import plot_pacf



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Weekly Resampled Partial Autocorrelations', fontsize=20)



plot_pacf(data.Dew_Point_Avg.resample('W').mean(),ax=ax[0, 0],lags=480, title='Dew_Point_Avg',) #row=0, col=0

plot_pacf(data.Dew_Point_Max.resample('W').mean(), ax=ax[1, 0],lags=480, title='Dew_Point_Max') #row=1, col=0

plot_pacf(data.Dew_Point_Min.resample('W').mean(),ax=ax[2, 0],lags=480, title='Dew_Point_Min') #row=2, col=0

plot_pacf(data.Humidity_Max.resample('W').mean(),ax=ax[3, 0],lags=480, title='Humidity_Max') #row=3, col=0

plot_pacf(data.Humidity_Min.resample('W').mean(),ax=ax[0, 1],lags=480, title='Humidity_Min') #row=0, col=1

plot_pacf(data.Pressure_Max.resample('W').mean(),ax=ax[1, 1],lags=480, title='Pressure_Max') #row=1, col=1

plot_pacf(data.Pressure_Min.resample('W').mean(),ax=ax[2, 1],lags=480, title='Pressure_Min') #row=2, col=1

plot_pacf(data.Temperature_Avg.resample('W').mean(),ax=ax[3, 1],lags=480, title='Temperature_Avg') #row=3, col=1

plot_pacf(data.Temperature_Max.resample('W').mean(),ax=ax[0, 2],lags=480, title='Temperature_Max') #row=0, col=2

plot_pacf(data.Temperature_Min.resample('W').mean(),ax=ax[1, 2],lags=480, title='Temperature_Min') #row=1, col=2

plot_pacf(data.Wind_speed_Max.resample('W').mean(),ax=ax[2, 2],lags=480, title='Wind_speed_Max') #row=2, col=2

plot_pacf(data.Wind_speed_Min.resample('W').mean(),ax=ax[3, 2],lags=480, title='Wind_speed_Min') #row=3, col=2



plt.show()
# monthly resampled partial autocorrelation





fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Monthly Resampled Partial Autocorrelations', fontsize=20)



plot_pacf(data.Dew_Point_Avg.resample('M').mean(),ax=ax[0, 0],lags=120, title='Dew_Point_Avg',) #row=0, col=0

plot_pacf(data.Dew_Point_Max.resample('M').mean(), ax=ax[1, 0],lags=120, title='Dew_Point_Max') #row=1, col=0

plot_pacf(data.Dew_Point_Min.resample('M').mean(),ax=ax[2, 0],lags=120, title='Dew_Point_Min') #row=2, col=0

plot_pacf(data.Humidity_Max.resample('M').mean(),ax=ax[3, 0],lags=120, title='Humidity_Max') #row=3, col=0

plot_pacf(data.Humidity_Min.resample('M').mean(),ax=ax[0, 1],lags=120, title='Humidity_Min') #row=0, col=1

plot_pacf(data.Pressure_Max.resample('M').mean(),ax=ax[1, 1],lags=120, title='Pressure_Max') #row=1, col=1

plot_pacf(data.Pressure_Min.resample('M').mean(),ax=ax[2, 1],lags=120, title='Pressure_Min') #row=2, col=1

plot_pacf(data.Temperature_Avg.resample('M').mean(),ax=ax[3, 1],lags=120, title='Temperature_Avg') #row=3, col=1

plot_pacf(data.Temperature_Max.resample('M').mean(),ax=ax[0, 2],lags=120, title='Temperature_Max') #row=0, col=2

plot_pacf(data.Temperature_Min.resample('M').mean(),ax=ax[1, 2],lags=120, title='Temperature_Min') #row=1, col=2

plot_pacf(data.Wind_speed_Max.resample('M').mean(),ax=ax[2, 2],lags=120, title='Wind_speed_Max') #row=2, col=2

plot_pacf(data.Wind_speed_Min.resample('M').mean(),ax=ax[3, 2],lags=120, title='Wind_speed_Min') #row=3, col=2
# Quarterly resampled partial autocorrelation



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Quarterly Resampled Partial Autocorrelations', fontsize=20)



plot_pacf(data.Dew_Point_Avg.resample('Q').mean(),ax=ax[0, 0],lags=40, title='Dew_Point_Avg',) #row=0, col=0

plot_pacf(data.Dew_Point_Max.resample('Q').mean(),ax=ax[1, 0],lags=40, title='Dew_Point_Max') #row=1, col=0

plot_pacf(data.Dew_Point_Min.resample('Q').mean(),ax=ax[2, 0],lags=40, title='Dew_Point_Min') #row=2, col=0

plot_pacf(data.Humidity_Max.resample('Q').mean(),ax=ax[3, 0],lags=40, title='Humidity_Max') #row=3, col=0

plot_pacf(data.Humidity_Min.resample('Q').mean(),ax=ax[0, 1],lags=40, title='Humidity_Min') #row=0, col=1

plot_pacf(data.Pressure_Max.resample('Q').mean(),ax=ax[1, 1],lags=40, title='Pressure_Max') #row=1, col=1

plot_pacf(data.Pressure_Min.resample('Q').mean(),ax=ax[2, 1],lags=40, title='Pressure_Min') #row=2, col=1

plot_pacf(data.Temperature_Avg.resample('Q').mean(),ax=ax[3, 1],lags=40, title='Temperature_Avg') #row=3, col=1

plot_pacf(data.Temperature_Max.resample('Q').mean(),ax=ax[0, 2],lags=40, title='Temperature_Max') #row=0, col=2

plot_pacf(data.Temperature_Min.resample('Q').mean(),ax=ax[1, 2],lags=40, title='Temperature_Min') #row=1, col=2

plot_pacf(data.Wind_speed_Max.resample('Q').mean(),ax=ax[2, 2],lags=40, title='Wind_speed_Max') #row=2, col=2

plot_pacf(data.Wind_speed_Min.resample('Q').mean(),ax=ax[3, 2],lags=40, title='Wind_speed_Min') #row=3, col=2



plt.show()
# Yearly resampled partial autocorrelation



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))

fig.suptitle('Yearly Resampled Partial Autocorrelations', fontsize=20)



plot_pacf(data.Dew_Point_Avg.resample('A').mean(),ax=ax[0, 0],lags=10, title='Dew_Point_Avg',) #row=0, col=0

plot_pacf(data.Dew_Point_Max.resample('A').mean(),ax=ax[1, 0],lags=10, title='Dew_Point_Max') #row=1, col=0

plot_pacf(data.Dew_Point_Min.resample('A').mean(),ax=ax[2, 0],lags=10, title='Dew_Point_Min') #row=2, col=0

plot_pacf(data.Humidity_Max.resample('A').mean(),ax=ax[3, 0],lags=10, title='Humidity_Max') #row=3, col=0

plot_pacf(data.Humidity_Min.resample('A').mean(),ax=ax[0, 1],lags=10, title='Humidity_Min') #row=0, col=1

plot_pacf(data.Pressure_Max.resample('A').mean(),ax=ax[1, 1],lags=10, title='Pressure_Max') #row=1, col=1

plot_pacf(data.Pressure_Min.resample('A').mean(),ax=ax[2, 1],lags=10, title='Pressure_Min') #row=2, col=1

plot_pacf(data.Temperature_Avg.resample('A').mean(),ax=ax[3, 1],lags=10, title='Temperature_Avg') #row=3, col=1

plot_pacf(data.Temperature_Max.resample('A').mean(),ax=ax[0, 2],lags=10, title='Temperature_Max') #row=0, col=2

plot_pacf(data.Temperature_Min.resample('A').mean(),ax=ax[1, 2],lags=10, title='Temperature_Min') #row=1, col=2

plot_pacf(data.Wind_speed_Max.resample('A').mean(),ax=ax[2, 2],lags=10, title='Wind_speed_Max') #row=2, col=2

plot_pacf(data.Wind_speed_Min.resample('A').mean(),ax=ax[3, 2],lags=10, title='Wind_speed_Min') #row=3, col=2



# plt.show()
# using decomposition to inspect for trend and seasonality



from statsmodels.tsa.seasonal import seasonal_decompose



decomposed1 = seasonal_decompose(data.Dew_Point_Avg, freq=360)

decomposed2 = seasonal_decompose(data.Dew_Point_Max, freq=360)

decomposed3 = seasonal_decompose(data.Dew_Point_Min, freq=360)

decomposed4 = seasonal_decompose(data.Humidity_Max, freq=360)

decomposed5 = seasonal_decompose(data.Humidity_Min, freq=360)

decomposed6 = seasonal_decompose(data.Pressure_Max, freq=360)

decomposed7 = seasonal_decompose(data.Pressure_Min, freq=360)

decomposed8 = seasonal_decompose(data.Temperature_Avg, freq=360)

decomposed9 = seasonal_decompose(data.Temperature_Max, freq=360)

decomposed10 = seasonal_decompose(data.Temperature_Min, freq=360)

decomposed12 = seasonal_decompose(data.Wind_speed_Max, freq=360)

decomposed13 = seasonal_decompose(data.Wind_speed_Min, freq=360)
# ploting the decompositioned time series , we are interested in Temperature

from pylab import rcParams



# Temperature_Avg

rcParams['figure.figsize'] = 11, 15

fig = decomposed8.plot()

plt.show()
# Temperature_MAX

rcParams['figure.figsize'] = 11, 15

fig = decomposed9.plot()

plt.show()
# Temperature_MIN

rcParams['figure.figsize'] = 11, 15

fig = decomposed10.plot()

plt.show()
# Let's take the differences in all variables and plot them to get rid of seasonality



weather_diff = data.diff()

weather_diff.plot(subplots=True, figsize=(20,40), colormap='plasma');
# Let's create some boxplots to better see outliers in my two variables



fig, ax = plt.subplots(4, 3, sharex=True,figsize=(30,30))



ax[0, 0].boxplot(data.Dew_Point_Avg) #row=0, col=0

ax[1, 0].boxplot(data.Dew_Point_Max) #row=1, col=0

ax[2, 0].boxplot(data.Dew_Point_Min.resample('A').mean()) #row=2, col=0

ax[3, 0].boxplot(data.Humidity_Max) #row=3, col=0

ax[0, 1].boxplot(data.Humidity_Min) #row=0, col=1

ax[1, 1].boxplot(data.Pressure_Max) #row=1, col=1

ax[2, 1].boxplot(data.Pressure_Min) #row=2, col=1

ax[3, 1].boxplot(data.Temperature_Avg) #row=3, col=1

ax[0, 2].boxplot(data.Temperature_Max) #row=0, col=2

ax[1, 2].boxplot(data.Temperature_Min) #row=1, col=2

ax[2, 2].boxplot(data.Wind_speed_Max) #row=2, col=2

ax[3, 2].boxplot(data.Wind_speed_Min) #row=3, col=2



ax[0, 0].set_title('Dew_Point_Avg',size=20) #row=0, col=0

ax[1, 0].set_title('Dew_Point_Max',size=20) #row=1, col=0

ax[2, 0].set_title('Dew_Point_Min',size=20) #row=2, col=0

ax[3, 0].set_title('Humidity_Max',size=20) #row=3, col=0

ax[0, 1].set_title('Humidity_Min',size=20) #row=0, col=1

ax[1, 1].set_title('Pressure_Max',size=20) #row=1, col=1

ax[2, 1].set_title('Pressure_Min',size=20) #row=2, col=1

ax[3, 1].set_title('Temperature_Avg',size=20) #row=3, col=1

ax[0, 2].set_title('Temperature_Max',size=20) #row=0, col=2

ax[1, 2].set_title('Temperature_Min',size=20) #row=1, col=2

ax[2, 2].set_title('Wind_speed_Max',size=20) #row=2, col=2

ax[3, 2].set_title('Wind_speed_Min',size=20) #row=3, col=2



plt.show()
# Let's model future daily temperatures



from statsmodels.tsa.arima_model import ARIMA

model_1 = ARIMA(data['Temperature_Avg'], (2,1,1))

model_1 = model_1.fit(trend='nc')

model_1.summary()
model_1.plot_predict(100, 400, dynamic=False);
model_1.forecast(10)
model_1.aic, model_1.bic
weather_month = data.asfreq('M').dropna()

weather_annual = data.asfreq('A')
# Forcasting monthly

# Let's model future temperatures using monthly resampling



for p in range(6):

    for d in range(2):

        for q in range(4):

            try:

                model_2=ARIMA(weather_month['Temperature_Avg'],(p,d,q)).fit(transparams=True)



                x=model_2.aic



                x1= p,d,q

            except:

                pass

                # ignore the error and go on
model_2.summary()
model_2.plot_predict(100, 300, dynamic=False);
model_2.forecast(10)
model_2.aic, model_2.bic
# Forcasting yearly

# Let's model future temperatures using yearly resampling



for p in range(6):

    for d in range(2):

        for q in range(4):

            try:

                model_3=ARIMA(weather_annual['Temperature_Avg'],(p,d,q)).fit(transparams=True)



                x=model_3.aic



                x1= p,d,q

            except:

                pass

                # ignore the error and go on
model_3.summary()
model_3.plot_predict(2, 20, dynamic=False);
# it seems that there is no real trend to forecast here even though both my AIC and BIC scores have significantly reduced in model_3

# This may suggest that forcasting yearly Temperature_Avg is more predictable using yearly than with higher time series frequancies
# Let's one more model for Monthly Temperature_Avg



import warnings

import itertools

import statsmodels.api as sm



temp_avg = data['Temperature_Avg'].resample('MS').mean()



# Define the p, d and q parameters to take any value between 0 and 2

p = d = q = range(0, 2)



# Generate all different combinations of p, q and q triplets

pdq = list(itertools.product(p, d, q))



# Generate all different combinations of seasonal p, q and q triplets

seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]



print('Examples of parameter combinations for Seasonal ARIMA...')

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))

print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))

print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))
warnings.filterwarnings("ignore") # specify to ignore warning messages



for param in pdq:

    for param_seasonal in seasonal_pdq:

        try:

            mod = sm.tsa.statespace.SARIMAX(temp_avg,

                                            order=param,

                                            seasonal_order=param_seasonal,

                                            enforce_stationarity=False,

                                            enforce_invertibility=False)



            results = mod.fit()



            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))

        except:

            continue
mod = sm.tsa.statespace.SARIMAX(temp_avg,

                                order=(1, 1, 1),

                                seasonal_order=(0, 1, 1, 12),

                                enforce_stationarity=False,

                                enforce_invertibility=False)



results = mod.fit()



print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))

plt.show()
pred = results.get_prediction(start=('2011-01-01'), dynamic=False)

pred_ci = pred.conf_int()


ax = temp_avg['2008':].plot(label='observed')

pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(15, 12))



ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.2)



ax.set_xlabel('Date')

ax.set_ylabel('Avg Temp')

plt.legend()



plt.show()
y_forecasted = pred.predicted_mean

y_truth = temp_avg['2011-01-01':]



# Compute the mean square error

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
y_forecasted
from sklearn.metrics import r2_score

r2_score(y_truth, y_forecasted)
pred_dynamic = results.get_prediction(start=('2011-01-01'), dynamic=True, full_results=True)

pred_dynamic_ci = pred_dynamic.conf_int()
ax = temp_avg['2008':].plot(label='observed', figsize=(20, 15))

pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)



ax.fill_between(pred_dynamic_ci.index,

                pred_dynamic_ci.iloc[:, 0],

                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)



ax.fill_betweenx(ax.get_ylim(), pd.to_datetime('2011-01-01'), data.Temperature_Avg.index[-1],

                 alpha=.1, zorder=-1)



ax.set_xlabel('Date')

ax.set_ylabel('Avg Temp')



plt.legend()

plt.show()
# Extract the predicted and true values of our time series

y_forecasted = pred_dynamic.predicted_mean

y_truth = temp_avg['2011-01-01':]



# Compute the mean square error

mse = ((y_forecasted - y_truth) ** 2).mean()

print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))
r2_score(y_truth, y_forecasted)
# Get forecast 500 steps ahead in future

pred_uc = results.get_forecast(steps=500)



# Get confidence intervals of forecasts

pred_ci = pred_uc.conf_int()
ax = temp_avg.plot(label='observed', figsize=(20, 15))

pred_uc.predicted_mean.plot(ax=ax, label='Forecast')

ax.fill_between(pred_ci.index,

                pred_ci.iloc[:, 0],

                pred_ci.iloc[:, 1], color='k', alpha=.25)

ax.set_xlabel('Date')

ax.set_ylabel('Avg Temp')



plt.legend()

plt.show()
# Could this suggest global warming ?!