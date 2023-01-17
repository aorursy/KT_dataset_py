import pandas as pd

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

print("Setup Complete")
# Set up code checking

from learntools.core import binder

binder.bind(globals())

from learntools.data_viz_to_coder.ex7 import *

print("Setup Complete")
# Check for a dataset with a CSV file

step_1.check()
# Fill in the line below: Specify the path of the CSV file to read

my_filepath ='../input/smart-home-dataset-with-weather-information/HomeC.csv'

# Check for a valid filepath to a CSV file in a dataset

step_2.check()
# Fill in the line below: Read the file into a variable my_data

my_data = pd.read_csv(my_filepath  ,   parse_dates=True)

home_dat = my_data.select_dtypes(exclude=['object'])



# you can convert a time from unix epoch timestamp to normal stamp using 

# import time 

# print( ' start ' , time.strftime('%Y-%m-%d %H:%S', time.localtime(1451624400)))





time_index = pd.date_range('2016-01-01 05:00', periods=503911,  freq='min')  

time_index = pd.DatetimeIndex(time_index)

home_dat = home_dat.set_index(time_index)

# Check that a dataset has been uploaded into my_data

step_3.check()
energy_data = home_dat.filter(items=[ 'gen [kW]', 'House overall [kW]', 'Dishwasher [kW]',

                                     'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 'Fridge [kW]',

                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]', 'Living room [kW]', 'Solar [kW]'])



weather_data = home_dat.filter(items=['temperature',

                                      'humidity', 'visibility', 'apparentTemperature', 'pressure',

                                      'windSpeed', 'windBearing', 'dewPoint'])
energy_data.head(10)
weather_data.head()
# Print the first five rows of the data

energy_per_day = energy_data.resample('D').sum()

energy_per_day.head()
energy_per_month = energy_data.resample('M').sum() # for energy we use sum to calculate overall consumption in period 

plt.figure(figsize=(20,10))

sns.lineplot(data= energy_per_month.filter(items=[ 'Dishwasher [kW]','House overall [kW]',

                                     'Furnace 1 [kW]', 'Furnace 2 [kW]', 'Home office [kW]', 'Fridge [kW]',

                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]', 'Living room [kW]', 'Solar [kW]']) , dashes=False  )

# use power == house overall

# gen power == solar 
plt.figure(figsize=(20,10))

# Plot the rooms consumption 

sns.lineplot(data= energy_per_month.filter(items=[      # remove the devices consumption 

                                     'Home office [kW]',

                                     'Wine cellar [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',

                                      'Living room [kW]']) , dashes=False  )
weather_per_day = weather_data.resample('D').mean()  # note!! (mean) # D =>> for day sample

weather_per_day.head()

weather_per_month = weather_data.resample('M').mean()                # M =>> for month sample




plt.figure(figsize=(20,8))



sns.lineplot(data= weather_per_month.filter(items=['temperature',

                                      'humidity', 'visibility', 'apparentTemperature',

                                      'windSpeed', 'dewPoint']) ,dashes=False )







weather_per_month.head()
plt.figure(figsize=(20,8))

sns.lineplot(data= energy_data.loc['2016-10-01 00:00' : '2016-10-01 23:00'].filter([ 'Home office [kW]',

                                     'Wine cellar [kW]', 'Garage door [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                 'Living room [kW]']),dashes=False , )
plt.figure(figsize=(20,8))

sns.lineplot(data= energy_data.loc['2016-10-01 00:00' : '2016-10-01 23:00'].filter([ 'Home office [kW]',

                                      'Garage door [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]',

                                 'Living room [kW]']),dashes=False , )
plt.figure(figsize=(20,8))

sns.lineplot(data= energy_per_day['Solar [kW]'],dashes=False , )
rooms_energy = energy_per_month.filter(items=[      # remove the devices consumption 

                                     'Home office [kW]',

                                     'Wine cellar [kW]', 'Kitchen 12 [kW]',

                                     'Kitchen 14 [kW]', 'Kitchen 38 [kW]', 'Barn [kW]',

                                      'Living room [kW]']) 

devices_energy = energy_per_month.filter(items=[ 'Dishwasher [kW]',

                                     'Furnace 1 [kW]', 'Furnace 2 [kW]',  'Fridge [kW]',

                                     'Garage door [kW]', 

                                     'Kitchen 38 [kW]', 'Barn [kW]', 'Well [kW]',

                                     'Microwave [kW]'])



all_rooms_consum = rooms_energy.sum()

all_devices_consum = devices_energy.sum()

print(all_rooms_consum)

print(all_devices_consum)

plot = all_rooms_consum .plot(kind = "pie", figsize = (5,5))

plot.set_title("Consumption for room")
plot = all_devices_consum .plot(kind = "pie", figsize = (5,5))

plot.set_title("Consumption for devices")
sns.regplot(x=energy_per_day['House overall [kW]'], y= weather_per_day['temperature'])
sns.regplot(x=energy_per_day.filter(items = ['Kitchen 12 [kW]','Kitchen 14 [kW]', 'Kitchen 38 [kW]']).sum(axis=1 ), y= weather_per_day['temperature'])
sns.regplot(x=energy_per_day['Fridge [kW]'], y= weather_per_day['temperature'])
sns.regplot(x=energy_per_day['Garage door [kW]'], y= weather_per_day['temperature'])