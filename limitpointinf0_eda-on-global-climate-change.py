import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))

pd.read_csv('../input/GlobalTemperatures.csv').tail()
import datetime
df = pd.read_csv('../input/GlobalTemperatures.csv')
df = df[['dt', 'LandAverageTemperature', 'LandAverageTemperatureUncertainty']]
df.columns = ['DT','LAT', 'LATU']
df['MIN'] = df.LAT - df.LATU
df['MAX'] = df.LAT + df.LATU
df.DT = pd.to_datetime(df.DT, format='%Y-%m-%d')
df['MONTH'] = df.DT.map(lambda x: x.month)
df['YEAR'] = df.DT.map(lambda x: x.year)
df = df.set_index(df.DT)
df.DT = pd.DatetimeIndex(df.DT).astype ( np.int64 )/1000000
df.head()
import seaborn as sns
sns.set_palette('Reds')
import matplotlib.pyplot as plt
sns.set(style='whitegrid')

t = '1900-01-01'
plt.figure(figsize=(20,10))
plt.title('Land Average Tempurature since {}'.format(t))
ax = sns.regplot(x="DT", y="LAT", data=df.loc[t:])
ax = sns.regplot(x="DT", y="MIN", data=df.loc[t:])
ax = sns.regplot(x="DT", y="MAX", data=df.loc[t:])
plt.show()
frames = [df[(df.YEAR >= 1900) & (df.YEAR <=1910)],df[(df.YEAR >= 2004) & (df.YEAR <= 2014)]]
result = pd.concat(frames)
result['PERIOD'] = result.YEAR.map(lambda x: '19\'s' if x < 2000 else '20\'s')

plt.figure(figsize=(15,5))
plt.title('LAT comparison between 1900-1910 and 2004-2014')
ax = sns.boxplot(x='PERIOD', y='LAT', data=result)
plt.show()
del result
lat = df[(df.YEAR >= 1900)].pivot("MONTH", "YEAR", "LAT")

plt.figure(figsize=(25,5))
plt.title("Heatmap beginning from 1990")
ax = sns.heatmap(lat)
plt.show()

del lat
grouped = df[df.YEAR > 1900].groupby('YEAR', as_index=False).mean().pivot('MONTH','YEAR','LAT')

plt.figure(figsize=(20,5))
plt.title("Heatmap beginning from 1990 (binned by year)")
ax = sns.heatmap(grouped, yticklabels=False)
plt.ylabel('')
plt.show()
del grouped