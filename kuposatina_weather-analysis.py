# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import seaborn as sns
import matplotlib.pyplot as plt
sns.set()
# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
data = pd.read_csv(r'../input/zelenogorsk-weather-19982020/weather_data_zelenogorsk.csv', index_col='date')
data.head()
data.index = pd.to_datetime(data.index)
fig, ax = plt.subplots(8,1, figsize=(15,44))
ax[0].plot(data['1998':'2000'])
ax[1].plot(data['2000':'2002'])
ax[2].plot(data['2003':'2005'])
ax[3].plot(data['2006':'2008'])
ax[4].plot(data['2009':'2011'])
ax[5].plot(data['2012':'2014'])
ax[6].plot(data['2015':'2017'])
ax[7].plot(data['2018':'2020']);
data_daymean = data.groupby(data.index.date).mean()
data_daymean.index = pd.to_datetime(data_daymean.index)
plt.figure(figsize=(20,7))
plt.plot(data_daymean)
plt.title('Daymean temperature');
data_interpol = data_daymean.interpolate(method='linear')
data_interpol.index = pd.to_datetime(data_interpol.index)

fig, ax = plt.subplots(1,1, figsize=(20,7))

g_data_interpol = data_interpol.groupby(data_interpol.index.month)

ax.plot(g_data_interpol.quantile(0.5), marker='o', color='blue', label='50%')

ax.fill_between(range(1,13), g_data_interpol.min().to_numpy().reshape(-1),
                g_data_interpol.max().to_numpy().reshape(-1), alpha=0.5, color='#bbe1fa', label='minmax')

ax.fill_between(range(1,13), g_data_interpol.quantile(0.1).to_numpy().reshape(-1),
                g_data_interpol.quantile(0.9).to_numpy().reshape(-1), alpha=0.2, color='#3282b8', label='90%')
ax.fill_between(range(1,13), g_data_interpol.quantile(0.25).to_numpy().reshape(-1),
                g_data_interpol.quantile(0.75).to_numpy().reshape(-1), alpha=0.4, color='#0f4c75', label='75%')

#plt.xticks(range(1998, 2020));
ax.set_xticks(range(1,13))
ax.set_xticklabels(data_interpol.index.month_name().unique().to_numpy(), rotation=0);

ax.set_yticks(range(-50,40,5))
ax.set_ylim(-60,40)
ax.legend()
ax.set_ylabel('Temperature')
ax.set_title('Temperature spread');
data_summer_15 = data_interpol[data_interpol >= 15]
data_summer_15.index = pd.to_datetime(data_summer_15.index)

data_summer_10 = data_interpol[data_interpol >= 10]
data_summer_10.index = pd.to_datetime(data_summer_10.index)

data_summer_20 = data_interpol[data_interpol >= 20]
data_summer_20.index = pd.to_datetime(data_summer_20.index)

summer_days_15 = []
summer_days_10 = []
summer_days_20 = []

for y in range(1998, 2020):
    #print('Count summer days {}: {}'.format(y, data_summer[data_summer.index.year == y].count()))
    summer_days_15.append(data_summer_15[data_summer_15.index.year == y].count())
    summer_days_10.append(data_summer_10[data_summer_10.index.year == y].count())
    summer_days_20.append(data_summer_20[data_summer_20.index.year == y].count())

fig, ax = plt.subplots(1,1, figsize=(10,7))
ax.plot(range(1998, 2020), summer_days_20, marker='v', color='#0f4c75', label='+20')
ax.plot(range(1998, 2020), summer_days_15, marker='o', color='#3282b8', label='+15')
ax.plot(range(1998, 2020), summer_days_10, marker='^', color='#bbe037', label='+10')
ax.set_xticks(range(1998, 2020));
ax.set_xticklabels(range(1998, 2020), rotation=45);
ax.set_ylabel('Days')
ax.set_title('Amount of days above 10, 15 and 20 degrees celsius')
ax.legend();
data_winter_15 = data_interpol[data_interpol <= -15]
data_winter_15.index = pd.to_datetime(data_winter_15.index)

data_winter_10 = data_interpol[data_interpol <= -10]
data_winter_10.index = pd.to_datetime(data_winter_10.index)

data_winter_20 = data_interpol[data_interpol <= -20]
data_winter_20.index = pd.to_datetime(data_winter_20.index)

winter_days_15 = []
winter_days_10 = []
winter_days_20 = []

for y in range(1998, 2021):
    #print('Count summer days {}: {}'.format(y, data_summer[data_summer.index.year == y].count()))
    winter_days_15.append(data_winter_15[data_winter_15.index.year == y].count())
    winter_days_10.append(data_winter_10[data_winter_10.index.year == y].count())
    winter_days_20.append(data_winter_20[data_winter_20.index.year == y].count())

fig, ax = plt.subplots(1,1, figsize=(10,7))
ax.plot(range(1998, 2021), winter_days_20, marker='v',color='#0f4c75', label='-20')
ax.plot(range(1998, 2021), winter_days_15, marker='o',color='#3282b8', label='-15')
ax.plot(range(1998, 2021), winter_days_10, marker='^',color='#bbe037', label='-10')
ax.set_xticks(range(1998, 2021));
ax.set_xticklabels(range(1998, 2021), rotation=45);
ax.set_ylabel('Days')
ax.set_title('Amount of days below 10, 15 and 20 degrees celsius')
ax.legend()
