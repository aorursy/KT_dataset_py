!pip install seaborn==0.11.0
import numpy as np

import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None



plt.rcParams['figure.figsize'] = [6, 4]

plt.rcParams['figure.dpi'] = 100 # 200 e.g. is really fine, but slower



sns.set(style="whitegrid")

sns.set_color_codes("pastel")
data = pd.read_csv("../input/climate-change-earth-surface-temperature-data/GlobalLandTemperaturesByCity.csv")

data.tail()
print("Total rows in the dataset:",data.count()["dt"])

print("Maximum recorded temperature:",data.max()["AverageTemperature"])

print("Minimum recorded temperature:",data.min()["AverageTemperature"])
city_data = data[data["City"]=="Guwahati"]

city_data.head()
print("Total rows in the dataset:",city_data.count()["dt"])

print("Maximum recorded temperature:",city_data.max()["AverageTemperature"])

print("Minimum recorded temperature:",city_data.min()["AverageTemperature"])

print("Average recorded temperature:",city_data.mean()["AverageTemperature"])
city_data.describe(include=['object'])
city_data = city_data.drop(["City","Country","Latitude","Longitude"],axis=1)

city_data.head()
city_data[['year', 'month', 'date']] = city_data.dt.str.split("-",expand=True)

city_data.head()
city_data.dtypes
city_data['dt'] = pd.to_datetime(city_data['dt'])

city_data['year'] = city_data['year'].astype(str).astype(int)

city_data['month'] = city_data['month'].astype(str).astype(int)

city_data['date'] = city_data['date'].astype(str).astype(int)

city_data.dtypes
city_data = city_data.drop(["dt","date"],axis=1)

city_data.head()
city_data.rename(columns = {'AverageTemperature':'temperature', 'AverageTemperatureUncertainty':'deviation'}, inplace = True)

city_data.head()
#city_data.reset_index(inplace=True) ## This resets the index to start from 0

city_data.index = np.arange(1, len(city_data) + 1) ## This resets the index to start from 0

city_data.head()
missing_data = city_data[city_data.isnull().any(axis=1)]

print("Total rows with missing data:",len(missing_data))
sns.countplot(x="year", data=missing_data)
sns.countplot(x="month", data=missing_data)
print("Number of rows before removing missing data:",len(city_data))

city_data = city_data.dropna()

print("Number of rows after removing missing data:",len(city_data))
sns.boxplot(x="month", y="temperature", data=city_data)
sns.lineplot(data=city_data, x="year", y="temperature")
sns.lineplot(data=city_data, x="year", y="temperature", estimator=np.median)
sns.lineplot(data=city_data[city_data["month"]==1], x="year", y="temperature")
sns.lineplot(data=city_data[city_data["month"]==6], x="year", y="temperature")
sns.lineplot(x="month", y="deviation", data=city_data)
sns.lineplot(x="year", y="deviation", data=city_data[city_data["month"]==1])
sns.lineplot(x="year", y="deviation", data=city_data[city_data["month"]==6])
sns.scatterplot(data=city_data, x="temperature", y="deviation")
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')

ax.scatter(city_data['month'], city_data['year'], city_data['temperature'], marker='o')

ax.set_xlabel('Month')

ax.set_ylabel('Year')

ax.set_zlabel('Temperature')

plt.show()