

import numpy as np

import pandas as pd

import datetime as dt # date and time processing functions

import matplotlib.pyplot as plt # basic plotting 

import matplotlib.dates as mdates # date processing in matplotlib

from matplotlib.offsetbox import AnchoredText

plt.style.use('ggplot') # use ggplot style
#import the data set

data_set=pd.read_csv('../input/GlobalLandTemperaturesByCountry.csv')

data_set.head(1000)



data_set['dt'] = pd.to_datetime(data_set['dt'])

data_set['month'] = data_set['dt'].dt.month



data_set['dt'] = pd.to_datetime(data_set['dt'])

data_set['year'] = data_set['dt'].dt.year



data_set['dt'] = pd.to_datetime(data_set['dt'])

data_set['day'] = data_set['dt'].dt.day



data_set['Date'] = pd.to_datetime(data_set[['year','month', 'day']])

data_set.index = data_set['Date'].values


plt.figure(figsize=(20,6))

plt.plot(data_set.year,data_set['AverageTemperature'], label='World Temp Over Time')


YearlyAveWorld = data_set.resample('12M').mean()

YearlyAveWorld.drop('month', axis = 1, inplace = True)

YearlyAveWorld.drop('day', axis = 1, inplace = True)

YearlyAveWorld = YearlyAveWorld[1:-1]



plt.figure(figsize=(20,6))

plt.plot(YearlyAveWorld.year,YearlyAveWorld['AverageTemperature'], label='Clearer Temp Over Time')
data_setPHI = data_set[data_set['Country'] == 'Philippines']

data_setIND = data_set[data_set['Country'] == 'Indonesia']

data_setTHA = data_set[data_set['Country'] == 'Thailand']

data_setVIE = data_set[data_set['Country'] == 'Vietnam']

data_setSIN = data_set[data_set['Country'] == 'Singapore']

data_setMAL = data_set[data_set['Country'] == 'Malaysia']

data_setCAM = data_set[data_set['Country'] == 'Cambodia']

data_setMYA = data_set[data_set['Country'] == 'Myanmar']

data_setLAO = data_set[data_set['Country'] == 'Laos']

data_setBRU = data_set[data_set['Country'] == 'Brunei']

data_setTIM = data_set[data_set['Country'] == 'Timor-Leste']

#merge all the narrowed down datasets

seasia = [data_setPHI, data_setIND, data_setTHA, data_setVIE, data_setSIN, data_setMAL, data_setCAM, data_setMYA, data_setLAO, data_setBRU, data_setTIM]

data_setSEA = pd.concat(seasia)
data_setSEA['dt'] = pd.to_datetime(data_setSEA['dt'])

data_setSEA['month'] = data_setSEA['dt'].dt.month



data_setSEA['dt'] = pd.to_datetime(data_setSEA['dt'])

data_setSEA['year'] = data_setSEA['dt'].dt.year



data_setSEA['dt'] = pd.to_datetime(data_setSEA['dt'])

data_setSEA['day'] = data_setSEA['dt'].dt.day



data_setSEA['Date'] = pd.to_datetime(data_setSEA[['year','month', 'day']])

data_setSEA.index = data_setSEA['Date'].values

YearlyAveSEA = data_setSEA.resample('12M').mean()

YearlyAveSEA.drop('month', axis = 1, inplace = True)

YearlyAveSEA.drop('day', axis = 1, inplace = True)

YearlyAveSEA = YearlyAveSEA[1:-1]



YearlyAvePHI = data_setPHI.resample('12M').mean()

YearlyAvePHI.drop('month', axis = 1, inplace = True)

YearlyAvePHI.drop('day', axis = 1, inplace = True)

YearlyAvePHI = YearlyAvePHI[1:-1]



#plot all three graphs

f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(20,15))

ax1.plot(YearlyAveWorld.year, YearlyAveWorld['AverageTemperature'])

ax2.plot(YearlyAveSEA.year, YearlyAveSEA['AverageTemperature'])

ax3.plot(YearlyAvePHI.year, YearlyAvePHI['AverageTemperature'])

plt.show()
start = 1853

end = dt.datetime.now().year + 1





f, ax = plt.subplots(figsize=(30,15))

month_fmt = mdates.DateFormatter('%b')

ax.xaxis.set_major_formatter(month_fmt)

ax.set_prop_cycle(plt.cycler('color', plt.cm.winter(np.linspace(0, 1, len(range(start, end))))))

ax.set_ylabel('Temperature C')

ax.set_xlabel('Month')

ax.set_title('Anual Change In Temperature');



for year in range(start, end):

    nyeardf = data_setPHI[['AverageTemperature', 'day', 'month']][data_setPHI['year'] == year]

    nyeardf['year'] = 1853

    nyeardf['Date'] = pd.to_datetime(nyeardf[['year','month','day']])

    nyeardf.index = nyeardf['Date'].values

    ax.plot(nyeardf.index,nyeardf['AverageTemperature'], label = year)