import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

plt.style.use('ggplot')

import datetime
data = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")

data.head()
data.info()
data.isnull().sum()
data_sample = data.sample(1000)



p = sns.PairGrid(data=data_sample, vars=['t1', 't2', 'hum', 'wind_speed', 'weather_code', 'is_holiday', 'is_weekend','season', 'cnt'])

p.map_diag(plt.hist)

p.map_offdiag(plt.scatter)
data['timestamp'] = data['timestamp'] .apply(lambda x :datetime.datetime.strptime(x,'%Y-%m-%d %H:%M:%S'))

data['month'] = data['timestamp'].apply(lambda x : str(x).split(' ')[0].split('-')[1])

data['day'] = data['timestamp'].apply(lambda x : str(x).split(' ')[0].split('-')[2])

data['hour'] = data['timestamp'].apply(lambda x : str(x).split(' ')[1].split(':')[0])
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

figure.set_size_inches(12, 8)



sns.boxplot(data=data, y='cnt', ax=ax1)

sns.boxplot(data=data, x='month', y='cnt', ax=ax2)

sns.boxplot(data=data, x='hour', y='cnt', ax=ax3)

sns.boxplot(data=data, x='day', y='cnt', ax=ax4)

fig,(ax1, ax2, ax3, ax4, ax5)= plt.subplots(nrows=5)

fig.set_size_inches(18,25)



sns.pointplot(data=data, x='hour', y='cnt', ax=ax1)

sns.pointplot(data=data, x='hour', y='cnt', hue='is_holiday', ax=ax2)

sns.pointplot(data=data, x='hour', y='cnt', hue='is_weekend', ax=ax3)

sns.pointplot(data=data, x='hour', y='cnt', hue='season', ax=ax4)

sns.pointplot(data=data, x='hour', y='cnt', hue='weather_code',ax=ax5)