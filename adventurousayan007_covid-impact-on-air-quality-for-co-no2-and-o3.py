#Source - https://www.kaggle.com/yuvrajmetrani/silkboard-bangalore-ambient-air-covid19lockdown
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline



import warnings

warnings.filterwarnings('ignore')
data1 = pd.read_csv("/kaggle/input/silkboard-bangalore-ambient-air-covid19lockdown/co_blr_aq_lockdown_effect_CSB.csv")

data2 = pd.read_csv("/kaggle/input/silkboard-bangalore-ambient-air-covid19lockdown/o3_blr_aq_lockdown_effect_CSB.csv")

data3 = pd.read_csv("/kaggle/input/silkboard-bangalore-ambient-air-covid19lockdown/no2_blr_aq_lockdown_effect_CSB.csv")

data4 = pd.read_csv("/kaggle/input/silkboard-bangalore-ambient-air-covid19lockdown/pm10_blr_aq_lockdown_effect_CSB.csv")

data5 = pd.read_csv("/kaggle/input/silkboard-bangalore-ambient-air-covid19lockdown/pm25_blr_aq_lockdown_effect_CSB.csv")

data6 = pd.read_csv("/kaggle/input/silkboard-bangalore-ambient-air-covid19lockdown/so2_blr_aq_lockdown_effect_CSB.csv")
data1.head()
data1.nunique()
data1.drop(['Unnamed: 0', 'location','city','country','parameter', 'unit', 'latitude', 'longitude', 'attribution','local'], axis=1, inplace=True)
data1.head()
data1['utc'] = pd.to_datetime(data1['utc'])
data1.head()
data1['date'] = data1['utc'].dt.strftime('%Y-%m-%d')
data1.head()
data1.tail()
df1 = data1.groupby(['date']).agg(mean_value = ('value', np.mean))

df1 = df1.reset_index()

df1.sort_values('date', ascending = True)
data1.isnull().sum()
plt.figure(figsize=(30,30))

sns.lineplot(x=df1.date, y= df1.mean_value, data=df1)

plt.xticks(rotation=45)

plt.show()
data1.dtypes
df2 = data1.groupby(data1.utc.dt.to_period("M")).agg('mean')

df2 = df2.reset_index()

df2.head()
df2['utc'] = df2['utc'].astype(str)

df2.dtypes
plt.figure(figsize=(20,20))

sns.lineplot(x=df2.utc, y= df2.value, data=df2)

plt.xticks(rotation=45)

plt.show()
df3 = data1.groupby(data1.utc.dt.to_period("W")).agg('mean')

df3 = df3.reset_index()

df3.head()

df3['utc'].unique()
data2.head()
data2.nunique()
data2.drop(['Unnamed: 0', 'location','city','country','parameter', 'unit', 'latitude', 'longitude', 'attribution','local'], axis=1, inplace=True)
data2.head()
data2.weekday.unique()
data2.tail()
data2['utc'] = pd.to_datetime(data2['utc'])
data2['date'] = data2['utc'].dt.strftime('%Y-%m-%d')
df4 = data2.groupby(data2.utc.dt.to_period("M")).agg('mean')

df4 = df4.reset_index()

df4.head()
df4['utc'] = df4['utc'].astype(str)

df4.dtypes
plt.figure(figsize=(20,20))

sns.lineplot(x=df4.utc, y= df4.value, data=df4)

plt.xticks(rotation=45)

plt.show()
data3.head()
data3.drop(['Unnamed: 0', 'location','city','country','parameter', 'unit', 'latitude', 'longitude', 'attribution','local'], axis=1, inplace=True)
data3['utc'] = pd.to_datetime(data3['utc'])



data3['date'] = data3['utc'].dt.strftime('%Y-%m-%d')



df5 = data3.groupby(data3.utc.dt.to_period("M")).agg('mean')

df5 = df5.reset_index()

df5.head()



df5['utc'] = df5['utc'].astype(str)

df5.dtypes



plt.figure(figsize=(20,20))

sns.lineplot(x=df5.utc, y= df5.value, data=df5)

plt.xticks(rotation=45)

plt.show()
column_names = ["CO", "O3", "NO2"]

df6 = pd.DataFrame(columns = column_names)
df2.value.values
df2.columns
df7 = pd.merge(df2,df4, on='utc')
df7.head()
df8 = pd.merge(df7,df5, on='utc')
df8.head()
df8.drop(['weekday_x','weekday_y'], axis=1, inplace=True)

df8.columns = ['utc', 'co','o3','no2']
df8.head()
plt.figure(figsize=(10,10))

sns.jointplot(x=df8.co, y= df8.o3, data=df8)

plt.xticks(rotation=45)

plt.show()
plt.figure(figsize=(10,10))

sns.scatterplot(x=df8.co, y= df8.o3, data=df8)

plt.xticks(rotation=45)

plt.show()
df8 = df8.set_index('utc')

df8.head()
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
df9 = pd.DataFrame(sc.fit_transform(df8), columns=df8.columns)

df9.head()

df9.index = df8.index

df9.head()
plt.figure(figsize=(16,9))

sns.lineplot(data=df9)

plt.xticks(rotation=45)

plt.show()