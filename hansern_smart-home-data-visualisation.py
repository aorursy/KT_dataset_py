## Pandas is a Python package providing fast, flexible, and expressive data structures designed to make working with “relational” or “labeled” data both easy and intuitive. It aims to be the fundamental high-level building block for doing practical, real world data analysis in Python.

import pandas as pd

## Matplotlib is a Python 2D plotting library which produces publication quality figures in a variety of hardcopy formats and interactive environments across platforms. Matplotlib can be used in Python scripts, the Python and IPython shells, the Jupyter notebook, web application servers, and four graphical user interface toolkits.

import matplotlib.pyplot as plt

## `%matplotlib` is a magic function in IPython. With this, the output of plotting commands is displayed inline within frontends like the Jupyter notebook, directly below the code cell that produced it. The resulting plots will then also be stored in the notebook document.

%matplotlib inline

## Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.

import seaborn as sns

import numpy as np

print("Setup Complete")
# Set up code checking

## The checking code and notebooks used in Kaggle Learn courses.

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Specify the path of the CSV file to read

my_filepath = '../input/smart-home-dataset-with-weather-information/HomeC.csv'

# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Read the data file into a variable my_data

## pandas.read_csv: Read a comma-separated values (csv) file into DataFrame.

my_data = pd.read_csv(my_filepath  ,   parse_dates=True)

my_data.info()
# Remove the type of data is object in the dataset.

home_dat = my_data.select_dtypes(exclude=['object'])



## you can convert a time from unix epoch timestamp to normal stamp using import time 

## print( ' start ' , time.strftime('%Y-%m-%d %H:%S', time.localtime(1451624400)))



# Data publisher says the dataset contains the readings with a time span of 1 minute of house appliances in kW from a smart meter and weather conditions of that particular region. So, I set freq='min' and convert Uinx time to readable date.

time_index = pd.date_range('2016-01-01 00:00', periods=503911, freq='min')

time_index = pd.DatetimeIndex(time_index)

home_dat = home_dat.set_index(time_index)

# Check that a dataset has been uploaded into my_data

step_3.check()
# Print the first 10 rows of the data

## pandas.DataFrame.head: This function returns the first n rows for the object based on position. It is useful for quickly testing if your object has the right type of data in it.

home_dat.head(10)
# Print the last 10 rows of the data

## This function returns last n rows from the object based on position. It is useful for quickly verifying data, for example, after sorting or appending rows.

home_dat.tail(10)
home_dat = home_dat[0:-1] ## == dataset[0:dataset.shape[0]-1] == dataset[0:len(dataset)-1] == dataset[:-1]

home_dat.tail()
# Separate two different Attributes

energy_data = home_dat.filter(items=['use [kW]', 'gen [kW]', 'House overall [kW]', 

                                     'Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 

                                     'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]', 

                                     'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]', 

                                     'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'])



weather_data = home_dat.filter(items=['temperature','humidity', 'apparentTemperature'])
# Print the first 5 rows of the energy data

energy_data.head()
# Print the first 5 rows of the weather data

weather_data.head()
# Genetate data per day

## pandas.DataFrame.resample: Convenience method for frequency conversion and resampling of time series.

energy_per_day = energy_data.resample('D').sum() # for energy we use sum to calculate overall consumption in period

energy_per_day.head()
fig, axes = plt.subplots(nrows=2, ncols=1)

energy_per_day['use [kW]'].plot(ax=axes[0],figsize=(20,10))

energy_per_day['House overall [kW]'].plot(ax=axes[1],figsize=(20,10))
energy_data = energy_data.drop(columns=['use [kW]'])

energy_per_day = energy_per_day.drop(columns=['use [kW]'])
fig, axes = plt.subplots(nrows=2, ncols=1)

energy_per_day['gen [kW]'].plot(ax=axes[0],figsize=(20,10))

energy_per_day['Solar [kW]'].plot(ax=axes[1],figsize=(20,10))
energy_data = energy_data.drop(columns=['gen [kW]'])

energy_per_day = energy_per_day.drop(columns=['gen [kW]'])
# Set the width and height of the figure

plt.figure(figsize=(20,10))



# Add title

plt.title("Overall energy consumption per day")

sns.lineplot(data = energy_per_day.filter(items=['House overall [kW]']), dashes=False)
energy_per_month = energy_data.resample('M').sum()

plt.figure(figsize=(20,10))

plt.title("Overall energy consumption per month")

sns.lineplot(data = energy_per_month.filter(items=['House overall [kW]']), dashes=False)

# use power == house overall

# gen power == solar
plt.figure(figsize=(20,10))

plt.title("Each appliance energy consumption per day")

sns.lineplot(data = energy_per_day.filter(items=['Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 

                                     'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]', 

                                     'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]', 

                                     'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]', 'Living room [kW]']), dashes=False)
plt.figure(figsize=(20,10))

plt.title("Each appliance energy consumption per month")

sns.lineplot(data = energy_per_month.filter(items=['Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 

                                     'Home office [kW]', 'Fridge [kW]', 'Wine cellar [kW]', 

                                     'Garage door [kW]', 'Kitchen 12 [kW]', 'Kitchen 14 [kW]', 

                                     'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]', 'Living room [kW]']), dashes=False)
energy_per_month.head(12)
plt.figure(figsize=(20,10))

plt.title("Devices energy consumption")



# Plot the devices consumption

sns.lineplot(data = energy_per_month.filter(items=['Dishwasher [kW]', 'Furnace 1 [kW]', 'Furnace 2 [kW]', 

                                     'Fridge [kW]', 'Garage door [kW]', 'Well [kW]',

                                     'Microwave [kW]']), dashes=False)
plt.figure(figsize=(20,10))

plt.title("Rooms energy consumption")



# Plot the rooms consumption 

sns.lineplot(data = energy_per_month.filter(items=[      # remove the devices consumption 

                                     'Home office [kW]', 'Wine cellar [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',

                                      'Living room [kW]']) , dashes=False)
plt.figure(figsize=(20,7))

plt.title("Solar generation per month")

sns.lineplot(data = energy_per_day.filter(['Solar [kW]']).resample('M').sum(),dashes=False)
plt.figure(figsize=(20,10))

plt.title("Home activity in day 2016-10-4")

sns.lineplot(data = energy_data.loc['2016-10-04 00:00' : '2016-10-04 23:59'].filter(['Home office [kW]', 

                                     'Wine cellar [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',

                                     'Living room [kW]']),dashes=False)
weather_per_day = weather_data.resample('D').mean()  # note!! (mean) # D =>> for day sample

weather_per_day.head()
weather_per_month = weather_data.resample('M').mean()                # M =>> for month sample

plt.figure(figsize=(15,5))

plt.ylabel('°F')

plt.title("Temperature mean per month")

sns.lineplot(data = weather_per_month.filter(items=['temperature', 'apparentTemperature']),dashes=False)
weather_per_month = weather_data.resample('M').mean()                # M =>> for month sample

plt.figure(figsize=(15,5))

plt.title("Humidity mean per month")

sns.lineplot(data = weather_per_month.filter(items=['humidity']),dashes=False)
rooms_energy = energy_per_month.filter(items=[      # remove the devices consumption 

                                     'Home office [kW]', 'Wine cellar [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',

                                     'Living room [kW]']) 

devices_energy = energy_per_month.filter(items=[    # remove the rooms consumption

                                     'Dishwasher [kW]',

                                     'Furnace 1 [kW]', 'Furnace 2 [kW]',  'Fridge [kW]',

                                     'Garage door [kW]', 'Well [kW]',

                                     'Microwave [kW]'])



all_rooms_consum = rooms_energy.sum()

all_devices_consum = devices_energy.sum()

print(all_rooms_consum)

print(all_devices_consum)
plot = all_rooms_consum.plot(kind="pie", autopct='%.2f', figsize=(10,10))

plot.set_title("Consumption for rooms")

plot.set_ylabel('%')
plot = all_devices_consum.plot(kind="pie", autopct='%.2f', figsize=(10,10))

plot.set_title("Consumption for devices")

plot.set_ylabel('%')
sns.regplot(x = energy_per_day['Furnace 2 [kW]'], y = weather_per_day['temperature'])
sns.regplot(x = energy_per_day['Wine cellar [kW]'], y = weather_per_day['temperature'])
sns.regplot(x = energy_per_day['Fridge [kW]'], y = weather_per_day['temperature'])
sns.regplot(x = energy_per_day['Barn [kW]'], y = weather_per_day['temperature'])