import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

import matplotlib.pyplot as plt



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_summary = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv')

df_item = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv')

df_station = pd.read_csv('../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv')
df_summary.head()
date_time = df_summary['Measurement date'].str.split(" ", n=1, expand=True)

date_time.head()
df_summary['date'] = date_time[0]

df_summary['time'] = date_time[1]

df_summary = df_summary.drop(['Measurement date'], axis=1)

df_summary.head()
df_seoul = df_summary.groupby(['date'], as_index=False).agg({'SO2':'mean', 'NO2':'mean', 'O3':'mean', 'CO':'mean', 'PM10':'mean', 'PM2.5':'mean'})

df_seoul.head()
df_seoul.plot(x='date')
corr = df_seoul.corr()

f, ax = plt.subplots(figsize=(11, 9))

cmap = sns.diverging_palette(220, 10, as_cmap=True)

sns.heatmap(corr, cmap=cmap, vmax=1, center=0,

            square=True, linewidths=.5, cbar_kws={"shrink": .5})
df_seoul['PM10_class'] = -1

df_seoul.head()
for (idx, row) in df_seoul.iterrows():

    pm10 = row[5]

    _class = -1

    if pm10 < 0:

        continue

    elif pm10 < 30:

        _class = 0

    elif pm10 < 80:

        _class = 1

    elif pm10 < 150:

        _class = 2

    else:

        _class = 3

    df_seoul.loc[idx, 'PM10_class'] = _class

df_seoul.head()
df_seoul['PM10_class'].value_counts().plot(kind="bar")
sns.jointplot(x=df_seoul["CO"], y=df_seoul["NO2"], kind='kde', xlim=(0,1),ylim=(0,0.13), color='g')

plt.show()