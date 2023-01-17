### LOAD LIBRARIES ###

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-poster')



### LOAD DATA ###

df = pd.read_csv('../input/GlobalLandTemperaturesByState.csv')



### SELECT CHINA ###

df_china = df[df.Country == 'China']

# Select columns

df_china = df_china[['dt', 'AverageTemperature']]

# Resampling to annual frequency

df_china.dt = pd.to_datetime(df_china.dt)

df_china.index = df_china.dt

df_china = df_china.resample('A-DEC').mean()

# From 1850 to 2013

df_china = df_china[df_china.index >= '1850-12-31']

# Create trend

z = np.polyfit(range(0, len(df_china.index)), df_china.AverageTemperature.tolist(), 2)

p = np.poly1d(z)

df_china['China'] = p(range(0, len(df_china.index)))

df_china = df_china.drop(['AverageTemperature'], 1)



### SELECT RUSSIA ###

df_rus = df[df.Country == 'Russia']

# Select columns

df_rus = df_rus[['dt', 'AverageTemperature']]

# Resampling to annual frequency

df_rus.dt = pd.to_datetime(df_rus.dt)

df_rus.index = df_rus.dt

df_rus = df_rus.resample('A-DEC').mean()

# From 1850 to 2013

df_rus = df_rus[df_rus.index >= '1850-12-31']

# Create trend

z = np.polyfit(range(0, len(df_rus.index)), df_rus.AverageTemperature.tolist(), 2)

p = np.poly1d(z)

df_rus['Russia'] = p(range(0, len(df_rus.index)))

df_rus = df_rus.drop(['AverageTemperature'], 1)



### SELECT USA ###

df_usa = df[df.Country == 'United States']

# Select columns

df_usa = df_usa[['dt', 'AverageTemperature']]

# Resampling to annual frequency

df_usa.dt = pd.to_datetime(df_usa.dt)

df_usa.index = df_usa.dt

df_usa = df_usa.resample('A-DEC').mean()

# From 1850 to 2013

df_usa = df_usa[df_usa.index >= '1850-12-31']

# Create trend

z = np.polyfit(range(0, len(df_usa.index)), df_usa.AverageTemperature.tolist(), 2)

p = np.poly1d(z)

df_usa['USA'] = p(range(0, len(df_usa.index)))

df_usa = df_usa.drop(['AverageTemperature'], 1)



### PLOT ###

plt.figure(figsize=[15,7])

df_china.China.plot(label='China')

df_rus.Russia.plot(label='Russia')

df_usa.USA.plot(label='USA')

plt.legend()

plt.title('Temperature Trends in China, Russia, USA (1850-2013)')

plt.ylabel('mean temperature by years')

plt.xlabel('')

plt.tight_layout()

plt.show()