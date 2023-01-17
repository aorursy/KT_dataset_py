import numpy as np

import pandas as pd

import matplotlib, matplotlib.pyplot as plt



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



data16 = pd.read_csv("../input/TimeSeries_TotalSolarGen_and_Load_IT_2016.csv")    

print(data16.shape)

data16.head(10)
data16.tail(10)
data16.columns = ['date_time', 'load', 'solar_gen'] # rename the columns

data16['date_time'] = pd.to_datetime(data16['date_time']) # new timestamp format

data16['date'] = data16['date_time'].dt.date # return date from timestamp

data16['time'] = data16['date_time'].dt.time # return time from timestamp

data16.head()
data16.drop(['date_time'], axis = 1, inplace = True) # remove column labeled date_time in the same dataframe (inplace)

data16 = data16[['date', 'time', 'load', 'solar_gen']] # reorder the columns

data16.head()
data16 = data16.pivot(index = 'time', columns = 'date')

print(data16.shape)
data16.head()
plt.figure() # colormap plot of solar power generation for each day, 1-hour period

plt.imshow(data16['solar_gen'], aspect = 'auto', interpolation = 'gaussian')

plt.yticks(np.arange(23, -1, -1))

plt.colorbar()

plt.xlabel('Day of year')

plt.ylabel('Time of day')

plt.title('Solar Generation [MW]')

plt.show()
plt.figure() # colormap plot of power demand for each day, 1-hour period

plt.imshow(data16['load'], aspect = 'auto', interpolation = 'gaussian')

plt.yticks(np.arange(23, -1, -1))

plt.colorbar()

plt.xlabel('Day of year')

plt.ylabel('Time of day')

plt.title('Load [MW]')

plt.show()
max_data16 = pd.DataFrame(data16['solar_gen'].max(), columns=['max_solar_gen']) # create a dataframe, 1 column

max_data16['max_load'] = data16['load'].max() # add a new column to the dataframe

max_data16.head()
plt.figure()

max_data16['max_solar_gen'].plot(style=['or-'])

plt.ylabel('Maximum daily Solar Generation [MW]')

plt.xlabel('Date')

plt.show()
fig, axes = plt.subplots(ncols=2, figsize=(15, 5))

axes[0].plot(max_data16['max_load'], 'og-')

axes[1].plot(max_data16['max_load'].iloc[:30], 'og-')

max_data16['weekday'] = max_data16.index.weekday

axes[1].plot(max_data16['max_load'].iloc[:30][max_data16['weekday']>=5], 'sb')

axes[0].set(title='Maximum daily load 2016', xlabel='Date', ylabel='Peak load [MW]')

axes[1].set(title='Maximum daily load January 2016', xlabel='Date', ylabel='Peak load [MW]')

plt.show()