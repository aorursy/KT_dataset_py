# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt

%config InlineBackend.figure_format = 'svg'

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import datetime

# Any results you write to the current directory are saved as output.
data_path = '../input/waves-measuring-buoys-data-mooloolaba/Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv'

data_waves = pd.read_csv(data_path, index_col="Date/Time", parse_dates=True)



data_waves.info()
data_waves.head(10)
data_waves.tail(10)
# Rename the data rows for convenience sake

data_waves = data_waves.rename(columns = {'Hs' : 'significant_wave_height' , 'Hmax' : 'maximum_wave_height', 'Tz' : 'zero_wave_period',

                       'Tp' : 'peak_wave_period' , 'SST' : 'sea_surface_temperature' , 'Peak Direction' : 'peak_direction'})

data_waves.describe().transpose()
sns.set_style("darkgrid")

"""

plt.figure(figsize = (12,6))

plt.title("Significant wave height")

sns.lineplot(data=data_waves)

plt.xlabel("Date")

"""
#Excluding entries after june 2019

data_waves = data_waves[(data_waves.index.year == 2019) & (data_waves.index.month > 6) == False]

data_waves.describe()



#Counting invalid entries for each data column

print("Counting invalid entries for each data column")

for heading in data_waves.columns:

    print(heading, 'contain', (data_waves[heading].values == -99.9).sum(), 'invalid entries')



plt.figure(figsize = (12,6))

plt.title("Wave characteristics")

sns.lineplot(data=data_waves)

plt.xlabel("Date")
data_waves_final1 = data_waves[['significant_wave_height', 'maximum_wave_height', 'zero_wave_period','peak_wave_period']]

data_waves_final1 = data_waves_final1[data_waves_final1['significant_wave_height'] != -99.9]



data_waves_final2 = data_waves[['peak_direction']]

data_waves_final2 = data_waves_final2[data_waves_final2['peak_direction'] != -99.9]



data_waves_final3 = data_waves[['sea_surface_temperature']]

data_waves_final3 = data_waves_final3[data_waves_final3['sea_surface_temperature'] != -99.9]



data_waves_final = data_waves[(data_waves['significant_wave_height'] != -99.9) & (data_waves['peak_direction'] != -99.9) & (data_waves['sea_surface_temperature'] != -99.9)]

data_waves_final.describe().transpose()
plt.figure(figsize = (12,6))

plt.title("Wave characteristics")

sns.lineplot(data=data_waves_final1)

plt.xlabel("Date")



plt.figure(figsize = (12,6))

plt.title("Peak direction")

sns.lineplot(data=data_waves_final2)

plt.xlabel("Date")



plt.figure(figsize = (12,6))

plt.title("Sea surface temperature")

sns.lineplot(data=data_waves_final3)

plt.xlabel("Date")
plt.figure(figsize = (8,4))

sns.kdeplot(data=data_waves_final1['significant_wave_height'], shade=True)

sns.kdeplot(data=data_waves_final1['maximum_wave_height'], shade=True)



plt.figure(figsize = (8,4))

sns.kdeplot(data=data_waves_final1['zero_wave_period'], shade=True)

sns.kdeplot(data=data_waves_final1['peak_wave_period'], shade=True)



plt.figure(figsize = (8,4))

sns.kdeplot(data=data_waves_final2['peak_direction'], shade=True)



plt.figure(figsize = (8,4))

sns.kdeplot(data=data_waves_final3['sea_surface_temperature'], shade=True)
plt.figure(figsize = (16,8))

sns.pairplot(data_waves_final)
plt.figure(figsize=(14,4))

sns.heatmap(data_waves_final1.corr(),annot=True,fmt='.3f',linewidths=2)

plt.show()
from scipy.stats import skew, kurtosis
# Find the skewness and Kurtosis on Wave Height Column

print("Skewness of the Waves Height : ", skew(data_waves['significant_wave_height']))

print("Kurtosis of the Waves Height : ", kurtosis(data_waves['significant_wave_height']))
plt.figure(figsize = (14,6))

plt.title("The peak energy wave period")

sns.lineplot(data=waves_data['Tp'])

plt.xlabel("Date")
plt.figure(figsize = (12,4))

plt.title("Direction (related to true north) from which the peak period waves are coming from")

sns.lineplot(data=waves_data['Peak Direction'])

plt.xlabel("Date")
plt.figure(figsize = (12,6))

plt.title("Approximation of sea surface temperature")

sns.lineplot(data=waves_data['SST'])

plt.xlabel("Date")