import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from datetime import datetime

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data1 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Generation_Data.csv")

data2 = pd.read_csv("/kaggle/input/solar-power-generation-data/Plant_1_Weather_Sensor_Data.csv")
print(data1.shape)

print(data2.shape)
data1.head()
data1.tail()
data2.head()
data2.tail()
data1.describe()
data1.groupby(['DATE_TIME'])['SOURCE_KEY'].count()
data2.head()
data1.dtypes
data2.dtypes
data1.isnull().sum()
data1.nunique()
#data1['DATE_TIME'] = datetime(data1['DATE_TIME'])

#data1['DATE_TIME'] = data1['DATE_TIME'].strptime('%Y-%m-%d %H:%M:%S')

data1['DATE_TIME'] = pd.to_datetime(data1['DATE_TIME'], infer_datetime_format=True)

data2['DATE_TIME'] = pd.to_datetime(data2['DATE_TIME'], infer_datetime_format=True)

data1.dtypes
data1.tail()
df = data1.groupby(['SOURCE_KEY']).agg(mean_daily_yield = ('DAILY_YIELD',np.mean), mean_total_yield = ('TOTAL_YIELD', np.mean)\

                                      , mean_ac_power = ('AC_POWER', np.mean), mean_dc_power = ('DC_POWER', np.mean) )

df.reset_index()

df.sort_values(['mean_ac_power', 'mean_dc_power'], ascending = [False,False])
plt.figure(figsize=(12,12))

sns.barplot(x = df.index, y = 'mean_daily_yield', data=df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(12,12))

sns.barplot(x = df.index, y = 'mean_total_yield', data=df)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(25,25))

sns.set(style="whitegrid")

sns.boxplot(x=data1.SOURCE_KEY, y=data1.DAILY_YIELD, data=data1)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(25,25))

sns.set(style="whitegrid")

sns.boxplot(x=data1.SOURCE_KEY, y=data1.TOTAL_YIELD, data=data1)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(25,25))

sns.set(style="whitegrid")

sns.scatterplot(x='mean_daily_yield', y='mean_total_yield', hue=df.index, data=df)

plt.xticks(rotation=45)

plt.show()
data1['DATE'] = data1['DATE_TIME'].dt.strftime('%Y-%m-%d')

data1.tail()
df2 = data1.groupby(['DATE','SOURCE_KEY']).agg(mean_daily_yield = ('DAILY_YIELD',np.mean), mean_total_yield = ('TOTAL_YIELD', np.mean)\

                                      , mean_ac_power = ('AC_POWER', np.mean), mean_dc_power = ('DC_POWER', np.mean) )

df2 = df2.reset_index()

df2.sort_values(['mean_ac_power', 'mean_dc_power'], ascending = [False,False], inplace= True)

df2.head(50)

df2.index
plt.figure(figsize=(30,30))

sns.barplot(x = df2.DATE, y = df2.mean_ac_power, data=df2, hue=df2.SOURCE_KEY)

plt.xticks(rotation=45)

plt.show()