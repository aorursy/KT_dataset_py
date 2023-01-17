import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

import seaborn as sns

print(os.listdir("../input"))
df = pd.read_csv("../input/GlobalLandTemperaturesByCity.csv")

df.set_index("Country",inplace = True)

Turkey_df = df.loc["Turkey"]

Turkey_df = Turkey_df.fillna(0)

Turkey_df['dt']=pd.to_datetime(Turkey_df['dt'], format='%Y%m%d', errors='ignore')

Turkey_df['YEAR']=pd.DatetimeIndex(Turkey_df['dt']).year

Turkey_df['MONTH']=pd.DatetimeIndex(Turkey_df['dt']).month

Turkey_df['DAY']=pd.DatetimeIndex(Turkey_df['dt']).day

columns = ["dt","AvgTemp","AvgTempUnc","City","Latitude","Longitude","Year","Month","Day"]

Turkey_df = pd.DataFrame(Turkey_df.values,columns=columns)

n_df = pd.concat([pd.DataFrame(Turkey_df["Year"].values.astype(int),columns = ["Year"]),

                  pd.DataFrame(Turkey_df["Month"].values.astype(int),columns=["Month"]),

                  pd.DataFrame(Turkey_df["Day"].values.astype(int),columns=["Day"]),

                  pd.DataFrame(Turkey_df["AvgTemp"].values.astype(float),columns=["AvgTemp"]),

                  pd.DataFrame(Turkey_df["City"].values,columns = ["City"]),

                  pd.DataFrame(Turkey_df["Latitude"].values,columns = ["Lat."]),

                  pd.DataFrame(Turkey_df["Longitude"].values,columns=["Long"])

                 ],axis = 1)
plt.figure(figsize = (20,6))

plt.title("Annual Avg.Temperature in Turkey")

sns.scatterplot(n_df["Year"],n_df["AvgTemp"],color = "black",marker = "+",estimator="median")

sns.lineplot(n_df["Year"],n_df["AvgTemp"],color = "red")

plt.show()
df_1950 = n_df[n_df["Year"] > 1950]

df_1950.head(3)
plt.figure(figsize = (20,5))

plt.title("Before 1950 for Avg.Temp in Turkey")

sns.scatterplot(df_1950["Year"],df_1950["AvgTemp"],markers = "+",color = "black")

sns.lineplot(df_1950["Year"],df_1950["AvgTemp"],color = "red")

plt.show()
n_df.head(3)

print("Maximum Avg.Temperature : " , n_df["AvgTemp"].max() , "\nMinimum AvgTemp : ", n_df["AvgTemp"].min().round())
#Turkey_df.drop(["dt"],axis = 1)
max_temp = n_df[n_df["AvgTemp"] > 32]

min_temp = n_df[n_df["AvgTemp"] < -14]

def season(month):

    if month >= 3 and month <= 5:

        return 'Spring'

    elif month >= 6 and month <= 8:

        return 'Summer'

    elif month >= 9 and month <= 11:

        return 'Autumn'

    else:

        return 'Winter'

n_df["Season"] = n_df["Month"].apply(season)
max_temp
min_temp
plt.figure(figsize = (20,5))

sns.lineplot(n_df["Season"],n_df["AvgTemp"],color = "red")
from mpl_toolkits.basemap import Basemap

import matplotlib.pyplot as plt

# setup Lambert Conformal basemap.

plt.figure(figsize = (20,10))

m = Basemap(width=11000000,height=8000000,projection='lcc',

            resolution=None,lat_1=26.,lat_2=45,lat_0=36,lon_0=42.)

m.shadedrelief()

lons = n_df["Long"].str.strip("E").values

lats = n_df["Lat."].str.strip("N").values

x,y = m(lons,lats)

plt.scatter(x,y,color = "black",marker = ".")

plt.show()
os.listdir("../input/")
global_temp = pd.read_csv("../input/GlobalTemperatures.csv")
global_temp["dt"] = pd.to_datetime(global_temp["dt"],format = "%Y%m%d",errors="ignore")

global_temp["Year"] = pd.DatetimeIndex(global_temp["dt"]).year

global_temp["Month"] = pd.DatetimeIndex(global_temp["dt"]).month

global_temp["Day"] = pd.DatetimeIndex(global_temp["dt"]).day
global_temp.tail(5)
global_df = pd.concat([pd.DataFrame(global_temp["Year"].values,columns = ["Year"]),

                      pd.DataFrame(global_temp["LandAverageTemperature"].values.astype(float),columns = ["LandAverageTemp"]),

                      pd.DataFrame(global_temp["LandMaxTemperature"].values.astype(float),columns = ["LandMaxTemp"]),

                      pd.DataFrame(global_temp["LandMinTemperature"].values.astype(float),columns=["LandMinTemp"]),

                      ],axis = 1)
global_df.head(5)
global_df.isnull().sum()
global_df = global_df.fillna(0)

global_df.head(5)
plt.figure(figsize = (20,5))

plt.title("Global Temperature")

sns.lineplot(global_df["Year"],global_df["LandAverageTemp"],color = "red",label = "AvgTemp.")

sns.lineplot(global_df["Year"],global_df["LandMaxTemp"],label = "MaxTemp")

sns.lineplot(global_df["Year"],global_df["LandMinTemp"],label = "MinTemp")

plt.grid()

plt.show()