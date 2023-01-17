# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# From time series data analysis using lstm tutorial
import sys
from scipy.stats import randint
import matplotlib.pyplot as plt # this is used for the plot the graph 
import seaborn as sns # used for plot interactive graph. 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))
din = pd.read_csv('../input/temp.over.cooktop.simple.csv', sep=',', 
                 parse_dates=True, 
                 low_memory=False, index_col='DateTime')
dout = pd.read_csv('../input/outdoor.hourly.temp.humidity.csv', sep=',', 
                 parse_dates=True, 
                 low_memory=False, index_col='DateTime')

#first the outdoor temperatures
dout.head(n=10)
# then the outdoor temperature and humidity
din.head(n=10)
din.sort_index(inplace=True)
din.head(n=10)
plt.rcParams['figure.figsize'] = 12, 3  # set default image size for plots 
din.tempC.plot(title='Temp Over Cooktop', color='red')
plt.show()
dout.AvgHourlyTempC.plot(title='Outdoor Temp', color='green')
plt.show()
dout[(dout['AvgHourlyTempC'] < -100)]
dout.loc[dout.AvgHourlyTempC < -100, 'AvgHourlyTempC'] = np.nan
dout.dropna(inplace=True)
dout.AvgHourlyTempC.plot(title='Outdoor Temp', color='green')
plt.show()
fig, axes = plt.subplots(nrows=2, ncols=1)
plt.rcParams['figure.figsize'] = 8,6
plt.subplots_adjust(hspace = .7)
din.plot(title='Indoor Temp', color='green', ax=axes[0])
dout.AvgHourlyTempC.plot(title='Outdoor Temp', color='red', ax=axes[1])
dout.describe()
din.describe()
din.tempC.plot(title='Cooktop Temp - Feb 1 to Feb 8', color='red')
plt.tight_layout()
plt.xlim('2018-02-01', '2018-02-08')
plt.show()
dout.AvgHourlyTempC.plot(title='Outdoor Temp - Feb 1 to Feb 8', color='blue')
plt.xlim('2018-02-01', '2018-02-08')
plt.show()
din15 = din.resample('15T').ffill()
din15.head(n=10)
din15.tempC.plot(title='Resampled Temp Over Cooktop - Nov 11 to 15', color='red')
plt.xlim('2017-11-11', '2017-11-15')
plt.show()
tmp = din.resample('1T').mean().interpolate()
din15mean = tmp.resample('15T').ffill()
din15mean.tempC.plot(title='Resampled and interpolated Temp Over Cooktop - Nov 11 to 15', color='red')
plt.xlim('2017-11-11', '2017-11-15')
plt.show()
