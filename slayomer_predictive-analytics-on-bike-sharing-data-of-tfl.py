import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sns

sns.set_style("darkgrid")
df = pd.read_csv("/kaggle/input/london-bike-sharing-dataset/london_merged.csv")

df.head()
df.timestamp = pd.to_datetime(df.timestamp)

df.timestamp[0].month
df.info()
df.describe()
df.timestamp.diff().mean()
df.timestamp.describe()
plt.figure(figsize=(10,7))

sns.distplot(df.cnt,label="Bike Share Distribution")

mean_share = df.cnt.mean()

plt.plot([mean_share,mean_share],[0,0.0012],"-",linewidth=5,label="Average # of Bike Share")

plt.text(mean_share+100,0.001201,int(mean_share),fontweight="bold",fontsize=12)

plt.legend()
df.groupby(pd.Grouper(key="timestamp",freq="1M")).sum()["cnt"].plot()
sns.pairplot(df.sample(frac=0.05))
corrs = abs(df.corr())



fig = plt.figure(figsize=(10,8))

sns.heatmap(corrs,annot=True)
weather_lookup = {

1 : "Clear",

2 : "Scattered clouds",

3 : "Broken clouds",

4 : "Cloudy",

7 : "Rain",

10 : "Rain with thunderstorm",

26 : "Snowfall",

94 : "Freezing Fog"}



weather_counts = df.weather_code.value_counts()

weather_counts.index = [weather_lookup[i] for i in weather_counts.index]

weather_counts.plot(kind="pie",autopct="%.0f%%")

plt.ylabel("")

plt.title("Weather Type Distribution",fontweight="bold")
plt.axhline(y=mean_share,linewidth=5,c="deepskyblue",label="Mean")

sns.barplot(x="is_holiday",y="cnt",data=df)

plt.legend()
temps = pd.qcut(df.t1,10)

sns.barplot(y=temps,x="cnt",data=df,orient="h")

plt.axvline(x=mean_share,label="Mean",c="deepskyblue")

plt.legend()
hums = pd.qcut(df.hum,10)

sns.barplot(y=hums,x="cnt",data=df,orient="h")

plt.axvline(x=mean_share,c="deepskyblue",label="Mean")

plt.legend()
winds = pd.qcut(df.wind_speed,10)

sns.barplot(y=winds,x="cnt",data=df,orient="h")

plt.axvline(x=mean_share,c="deepskyblue",label="Mean")

plt.legend()
#first replace values with the categorical names in the weather_code column

df.weather_code.replace({

1 : "Clear",

2 : "Scattered clouds",

3 : "Broken clouds",

4 : "Cloudy",

7 : "Rain",

10 : "Rain with thunderstorm",

26 : "Snowfall",

94 : "Freezing Fog"},inplace=True)



sns.barplot(y="weather_code",x="cnt",data=df,orient="h",

            order=["Clear","Scattered clouds","Broken clouds","Cloudy","Rain","Rain with thunderstorm","Snowfall","Freezing Fog"])

plt.axvline(x=mean_share,c="deepskyblue",label="Mean")

plt.legend()
df.is_holiday.replace({1:"Holiday",0:"Workday"},inplace=True)

sns.barplot(x="cnt",y="is_holiday",data=df,orient="h")

plt.axvline(x=mean_share,c="deepskyblue",label="Mean")

plt.legend()
df.is_weekend.replace({1:"Weekend",0:"Weekday"},inplace=True)

sns.barplot(x="cnt",y="is_weekend",data=df,orient="h")

plt.axvline(x=mean_share,c="deepskyblue",label="Mean")

plt.legend()
df.season.replace({0:"Spring",1:"Summer",2:"Fall",3:"Winter"},inplace=True)

sns.barplot(y="season",x="cnt",data=df,orient="h",order=["Spring","Summer","Fall","Winter"])



plt.axvline(x=mean_share,c="deepskyblue",label="Mean")

plt.legend()