import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates 

%matplotlib inline 

plt.style.use('fivethirtyeight')
# Now I'll just import the dataset using pandas

seaice = pd.read_csv('../input/seaice.csv')

# Get some Basic stats

seaice['Extent'].describe()

# Convert the Year, Month, Day columns to date format

seaice['Date'] = pd.to_datetime(seaice[['Year','Month','Day']])
# Group the dataset by year

iceyears = seaice.groupby('Year')

# Pull out mean of the 'Extent' variable

annualextent = iceyears.mean()['Extent']

# plot the average annual extent

plt.plot(annualextent)

plt.title('Average sea ice extent')

plt.ylabel(r'Area (*10$^6$ km$^2$)')

plt.xlabel('Year')
# Plot hemispheres separately

# Northern hemisphere df

northice = seaice[seaice['hemisphere']=='north']

# Southern hemisphere df

southice = seaice[seaice['hemisphere']=='south']



# Plot the annual maximum and minimum sea ice for each hemisphere

northice_years = northice.groupby('Year')

southice_years = southice.groupby('Year')



# Set up axes

f, axarr = plt.subplots(2, sharex=True, figsize=(7.5,6))

northmax = northice_years['Extent'].max()

northmin = northice_years['Extent'].min()

axarr[0].plot(northmax)

axarr[0].plot(northmin)

axarr[0].set_title('Northern hemisphere')



southmax = southice_years['Extent'].max()

southmin = southice_years['Extent'].min()

axarr[1].plot(southmax)

axarr[1].plot(southmin)

axarr[0].set_title('Southern hemisphere')