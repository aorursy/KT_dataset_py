import pandas as pd

climate_change = pd.read_csv('../input/climate-change-dataset-datacamp/climate_change.csv')

climate_change.describe



import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(climate_change.index, climate_change['co2'])

ax.set_xlabel('Time')

ax.set_ylabel('CO2 (ppm)')

plt.show()

# Let's observe the graph. Does it make sense?
# Explore a decade

import matplotlib.pyplot as plt



# Use plt.subplots to create fig and ax

fig, ax = plt.subplots()



# Create variable seventies with data from "1970-01-01" to "1979-12-31"

seventies = climate_change["1970-01-01" :"1979-12-31"]



# Add the time-series for "co2" data from seventies to the plot

ax.plot(seventies.index, seventies["co2"])



# Show the figure

plt.show()

climate_change = pd.read_csv('../input/climate-change-dataset-datacamp/climate_change.csv', parse_dates = ['date'], index_col = 'date')

# Note the parameters 'parse_dates' & 'index_col'

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

ax.plot(climate_change.index, climate_change['co2'], color = 'red')

ax.set_xlabel('Time')

ax.set_ylabel('Carbon di oxide')

plt.show()
import matplotlib.pyplot as plt



# Use plt.subplots to create fig and ax

fig, ax = plt.subplots()



# Create variable seventies with data from "1970-01-01" to "1979-12-31"

seventies = climate_change["1980-01-01" :"1989-12-31"]



# Add the time-series for "co2" data from seventies to the plot

ax.plot(seventies.index, seventies["co2"])



# Show the figure

plt.show()