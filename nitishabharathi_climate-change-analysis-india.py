import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

import warnings

warnings.filterwarnings('ignore')

%matplotlib inline



data = pd.read_csv('../input/daily-temperature-of-major-cities/city_temperature.csv')

data_india = data[data['Country'] == 'India']

data_india = data_india[data_india['Year'] <2020]
data_india = data[data['Country'] == 'India']

data_india = data_india[data_india['Year'] <2020]
def to_celsius(F):

    return round(5/9*(F - 32),1)



data_india['AvgTemperature'] = data_india['AvgTemperature'].apply(to_celsius)
plt.figure(figsize= (15,10))

sns.pointplot(x='Year', y='AvgTemperature', data=data_india);

plt.title('Average Temperature India - Yearly Trend',fontsize=20);
plt.figure(figsize= (15,10))

sns.pointplot(x='Month', y='AvgTemperature', data=data_india);

plt.title('Average Temperature India - Monthly Trend',fontsize=20);
plt.figure(figsize= (15,10))

sns.pointplot(x='Year', y='AvgTemperature', data=data_india,hue='City');

plt.title('Average Temperature India - Yearly Trend',fontsize=20);
plt.figure(figsize= (15,10))

sns.pointplot(x='Month', y='AvgTemperature', data=data_india,hue='City');

plt.title('Average Temperature India - Monthly Trend',fontsize=20);