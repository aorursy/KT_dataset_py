# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

from numpy.fft import fft, ifft

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

# Import data
global_temp = pd.read_csv('../input/GlobalTemperatures.csv', index_col='dt', parse_dates=True)

# Fill the gaps in the series
global_temp.fillna(method='ffill')

# Skip the first years and start the series at the beginning of spring,
# so seasonal variations can be captured
global_temp = global_temp['1753-03-21':]

# Plot initial data
plt.figure(figsize=(15,4))
global_temp['LandAverageTemperature'].plot()
plt.grid()
plt.show()
# Resample the series and visualise at different scales
plt.figure(figsize=(15,16))

# Seasonal
seasonal_temp = global_temp.resample('3M', how='mean')
plt.subplot(4,1,1)
seasonal_temp['LandAverageTemperature'].plot()
plt.ylim([0,18])
plt.grid()

# half year
bi_seasonal_temp = global_temp.resample('6M', how='mean')
plt.subplot(4,1,2)
bi_seasonal_temp['LandAverageTemperature'].plot()
plt.ylim([0,18])
plt.grid()

# Yearly
year_temp = global_temp.resample('A', how='mean')
plt.subplot(4,1,3)
year_temp['LandAverageTemperature'].plot()
plt.ylim([0,18])
plt.grid()

# 4-Yearly
year_4_temp = global_temp.resample('4A', how='mean')
plt.subplot(4,1,4)
year_4_temp['LandAverageTemperature'].plot()
plt.ylim([0,18])
plt.grid()
plt.show()
## eplore the autocorrelation of temperature data
lat = np.array(global_temp['LandAverageTemperature'])

# detrend the seasonal data by removing the average
det_lat = lat - np.average(lat)

# Get correlogram for 24 seasons (2 years)
seasonal_correlogram = [1.0, ]
seasonal_correlogram.extend([np.corrcoef(det_lat[:-i], det_lat[i:])[0, 1] for i in range(1, 25)])

plt.plot(seasonal_correlogram)
plt.grid()
plt.xlabel('Periods [Months]')
plt.ylabel('Correlation')
plt.title('Autocorrelation')
plt.show()

## Therefore cold winters are followed by hot summers, or hot summer followed by cold winters
# Analysing seasonal changes over time
seasonal_lat = np.array(seasonal_temp['LandAverageTemperature'])

# Parse into stations
spring = seasonal_lat[::4]
summer = seasonal_lat[1::4]
fall = seasonal_lat[2::4]
winter = seasonal_lat[3::4]


plt.figure(figsize=(12,3))
ax = plt.subplot(1,1,1)

box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])

plt.plot(spring, label='Spring')
plt.plot(summer, label='Summer')
plt.plot(fall, label='Fall')
plt.plot(winter, label='Winter')

plt.xlim([0, len(summer)])
plt.grid()
plt.xlabel('Year')
plt.ylabel('Average Temperature [C]')

plt.legend(bbox_to_anchor=(1.18, 1.04))
# Seasonal analysis
seasons = [spring, summer, fall, winter]
seasons_string = ['spring', 'summer', 'fall', 'winter']

# Detrend for each of the seasons
seasons_average = [np.average(season) for season in seasons]
seasons_det = [seasons[i] - seasons_average[i] for i in range(len(seasons))]

plt.figure(figsize=[12,6])
plt.subplot(2,1,1)
[plt.plot(seasons_det[i], label=seasons_string[i]) for i in range(len(seasons))]
plt.ylabel('Centered Temperature')
plt.grid()
plt.xlim([0, len(seasons_det[0])])


## do the regression analysis
# Get the fourier coefficients
seasons_fft = [fft(season) for season in seasons_det]

# Get the power spectrum
seasons_ps = [np.abs(season)**2 for season in seasons_fft]

plt.subplot(2,1,2)
[plt.plot(seasons_ps[i], label=seasons_string[i]) for i in range(len(seasons))]
plt.xlabel('Frequency [Months]')
plt.ylabel('Power spectrum')
plt.xlim([0, 30])
plt.grid()
plt.show()
## Clean each of the time series in the seasons by selecting such that the power spectrum is higher than 500
clean_seasons_ps = seasons_ps[:]
clean_seasons_ps = [[seasons_fft[season_i][year_i] if seasons_ps[season_i][year_i] > 500 else 0 
                     for year_i in range(len(seasons_fft[0]))] for season_i in range(len(seasons_ps))]

plt.figure(figsize=[12,9])
plt.subplot(3,1,1)
plt.plot(np.transpose(clean_seasons_ps))
plt.xlim([0, 30])
plt.grid()

## redraw the series only with significant harmonics
seasons_series_clean = [np.real(ifft(serie)) for serie in clean_seasons_ps]

plt.subplot(3,1,2)
[plt.plot(seasons_series_clean[i], label=seasons_string[i]) for i in range(len(seasons))]
plt.xlim([0, len(seasons_det[0])])
plt.legend(bbox_to_anchor=(1.18, 1.04))
plt.grid()

## put the trend back into the dataset
seasonal_trends = [seasons_series_clean[i] + seasons_average[i] for i in range(len(seasons))]

plt.subplot(3,1,3)
[plt.plot(seasonal_trends[i], label=seasons_string[i]) for i in range(len(seasons))]
plt.xlim([0, len(seasons_det[0])])
plt.legend(bbox_to_anchor=(1.18, 1.04))
plt.grid()
plt.show()