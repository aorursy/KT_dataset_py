import os

import sys

import pandas as pd

import matplotlib.pyplot as plt

from matplotlib.dates import DateFormatter
for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
def parse(x):

    return pd.to_datetime(x, infer_datetime_format=True, unit='s',utc=True)
nrows_read = 100000 # specify 'None' if want to read whole file (405,184 rows)

data_path = '/kaggle/input/environmental-sensor-data-132k/iot_telemetry_data.csv'



df = pd.read_csv(data_path,

                  delimiter=',',

                  nrows = nrows_read,

                  header=0,

                  infer_datetime_format=True,

                  date_parser=parse,

                  index_col=['ts'])
# sort data

df = df.sort_values(by='ts', ascending=True)
# convert celsius to fahrenheit (°C to °F)

df['temp'] = (df['temp'] * 1.8) + 32
# preview data

df.head(5)
# filter temp/humidity, by device, for outliers (>1% & <99%)

df = df.loc[df['temp'] >df.groupby('device').temp.transform(lambda x: x.quantile(.01))]

df = df.loc[df['temp'] < df.groupby('device').temp.transform(lambda x: x.quantile(.99))]



df = df.loc[df['humidity'] > df.groupby('device').humidity.transform(lambda x: x.quantile(.01))]

df = df.loc[df['humidity'] < df.groupby('device').humidity.transform(lambda x: x.quantile(.99))]
# group data by iot device

groups = df.groupby('device')
print('DataFrame Stats')

print('-------------')

print('Record count: {:,}'.format(df['temp'].count()))

print('DataFrame size (MB): {:,.2f}'.format(sys.getsizeof(df)/1024/1024))

print('-------------')

print('Time range (min): {:%Y-%m-%d %H:%M:%S %Z}'.format(df.index[1]))

print('Time range (max): {:%Y-%m-%d %H:%M:%S %Z}'.format(df.index[-1]))

print('Temperature (min): {:.2f}'.format(df['temp'].min()))

print('Temperature (max): {:.2f}'.format(df['temp'].max()))

print('Humidity (min): {:.2f}{}'.format(df['humidity'].min(), '%'))

print('Humidity (max): {:.2f}{}'.format(df['humidity'].max(), '%'))

print('-------------')

print('Record count:\n{}'.format(groups.size()))
_, ax = plt.subplots(1, 1, figsize=(18, 9))

for device, group in groups:

    ax.plot(group.temp,

            group.humidity,

            marker='o',

            linestyle='',

            alpha=.5,

            ms=10,

            label=device)

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Temperature vs. Humidity')

plt.xlabel('Temperature (˚F)')

plt.ylabel('Humidity (%)')

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(18, 9))

for device, group in groups:

    group.mean = group.temp.rolling(window=20).mean()

    ax.plot(group.mean,

            label=device)

fig.autofmt_xdate()

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Temperature Comparison over Time')

plt.ylabel('Temperature (˚F)')

plt.xlabel('Time')

plt.show()
fig, ax = plt.subplots(1, 1, figsize=(18, 9))

for device, group in groups:

    group.mean = group.humidity.rolling(window=20).mean()

    ax.plot(group.mean,

            label=device)

fig.autofmt_xdate()

ax.grid()

ax.margins(0.05)

ax.legend()

plt.title('Humidity Comparison over Time')

plt.ylabel('Humidity (%)')

plt.xlabel('Time')

plt.show()