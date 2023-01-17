# Imports

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# Data importations

light = pd.read_excel("../input/temperature-and-light-data/Light_measurement.xlsx", index_col = 0, parse_dates=True)

temperature = pd.read_excel("../input/temperature-and-light-data/Temperature_measurement.xlsx", index_col = 0, parse_dates=True)
# Light dataset split according to hour

l_first_day = light[(light.index > '2020-02-21 12:00:00')]

l_second_day = light[(light.index < '2020-02-21 12:00:00')]
# Second day's date change

l_second_day.index = l_second_day.index + pd.Timedelta(days=1)
# Whole lihgt dataset rebuild

light = pd.concat([l_first_day, l_second_day])
# Verification

light
# Same things with temperatures

t_first_day = temperature[(temperature.index > '2020-02-21 12:00:00')]

t_second_day = temperature[(temperature.index < '2020-02-21 12:00:00')]



t_second_day.index = t_second_day.index + pd.Timedelta(days=1)



temperature = pd.concat([t_first_day, t_second_day])



temperature
# Plotting



import matplotlib.pyplot as plt

import matplotlib.dates as mdates



fig, ax1 = plt.subplots(1)



# Light's plot

color = 'tab:green'

ax1.set_xlabel('time')

ax1.set_ylabel('light', color=color)

ax1.plot(light.index, light["Light"], color=color)

ax1.tick_params(axis='y', labelcolor=color)



# instantiate a second axes that shares the same x-axis

ax2 = ax1.twinx()



# Temp's plot

color = 'tab:red'

ax2.set_ylabel('light', color=color)  # we already handled the x-label with ax1

ax2.plot(temperature.index, temperature["Temperature"], color=color)

ax2.tick_params(axis='y', labelcolor=color)



# Dates on xaxis formatting

fig.autofmt_xdate()

xfmt = mdates.DateFormatter('%H:%M')

ax1.xaxis.set_major_formatter(xfmt)