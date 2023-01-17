import pandas as pd

%matplotlib inline

import seaborn as sns

import matplotlib.pyplot as plt
data = pd.read_csv("../input/london-bike-sharing-dataset/london_merged.csv")

print(data.shape)

data.head()
data.info()
data.dtypes
data["timestamp"]=pd.to_datetime(data["timestamp"])

data.dtypes
data["season"] = data["season"].replace(0,"spring").replace(1,"summer").replace(2,"autumn").replace(3,"winter")
data["is_holiday"] = data["is_holiday"].replace(0,"not holiday").replace(1,"holiday")
data["weather_code"] = data["weather_code"].replace(1,"Clear").replace(2,"scattered clouds / few clouds").replace(3,"Broken clouds").replace(4,"Cloudy").replace(7,"Rain/ light Rain shower/ Light rain").replace(10,"rain with thunderstorm").replace(26,"snowfall").replace(94,"Freezing Fog")
data["year"] = data["timestamp"].dt.year

data["month"] = data["timestamp"].dt.month

data["day"] = data["timestamp"].dt.day

data["hour"] = data["timestamp"].dt.hour

data["datetime-dayofweek"] = data["timestamp"].dt.dayofweek

data.head()
data.loc[data["datetime-dayofweek"] == 0, "weekday"] = "Monday"

data.loc[data["datetime-dayofweek"] == 1, "weekday"] = "Tuesday"

data.loc[data["datetime-dayofweek"] == 2, "weekday"] = "Wednesday"

data.loc[data["datetime-dayofweek"] == 3, "weekday"] = "Thursday"

data.loc[data["datetime-dayofweek"] == 4, "weekday"] = "Friday"

data.loc[data["datetime-dayofweek"] == 5, "weekday"] = "Saturday"

data.loc[data["datetime-dayofweek"] == 6, "weekday"] = "Sunday"

data.head()
column = ["year","month","day","hour","weekday","cnt","t1","t2","hum","wind_speed","weather_code","is_holiday","is_weekend","season"]

data = data[column]

data.head()
figure, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)



figure.set_size_inches(18, 8)



sns.barplot(data=data, x="year", y="cnt", ax=ax1)

sns.barplot(data=data, x="month", y="cnt", ax=ax2)

sns.barplot(data=data, x="day", y="cnt", ax=ax3)

sns.barplot(data=data, x="hour", y="cnt", ax=ax4)

plt.show()
figure, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(nrows=5, ncols=1)



figure.set_size_inches(18, 20)



sns.pointplot(data=data, x="hour", y="cnt", ax=ax1)

sns.pointplot(data=data, x="hour", y="cnt", hue="is_holiday", ax=ax2)

sns.pointplot(data=data, x="hour", y="cnt", hue="weekday", ax=ax3)

sns.pointplot(data=data, x="hour", y="cnt", hue="season", ax=ax4)

sns.pointplot(data=data, x="hour", y="cnt", hue="weather_code", ax=ax5)



plt.show()
plt.figure(figsize=[18,15])

sns.pointplot(data=data, x="hour", y="cnt", hue="weather_code")

plt.show()