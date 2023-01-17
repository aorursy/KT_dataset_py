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

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
#read in data

station_aqi = pd.read_csv("../input/air-quality-data-in-india/station_day.csv")
stations = pd.read_csv("../input/air-quality-data-in-india/stations.csv")
station_aqi.head()
station_aqi.shape
station_aqi["Date"] = pd.to_datetime(station_aqi["Date"])
import datetime as dt
station_aqi["Date"].dt.year.value_counts()
data_2015 = station_aqi.loc[station_aqi["Date"].dt.year == 2015]
data_2016 = station_aqi.loc[station_aqi["Date"].dt.year == 2016]
data_2017 = station_aqi.loc[station_aqi["Date"].dt.year == 2017]
data_2018 = station_aqi.loc[station_aqi["Date"].dt.year == 2018]
data_2019 = station_aqi.loc[station_aqi["Date"].dt.year == 2019]
data_2020 = station_aqi.loc[station_aqi["Date"].dt.year == 2020]
print (data_2015.shape, data_2016.shape, data_2017.shape, data_2018.shape, data_2019.shape, data_2020.shape)
def plot_pol(station_id, indicator):
    
    plt.figure(figsize=(10,7))
    
    if station_id == "All":
        
        sns.lineplot(x = station_aqi["Date"].dt.month, y= station_aqi[indicator], data = station_aqi, hue = station_aqi["Date"].dt.year, err_style = None)
    else:
        dataf = station_aqi[station_aqi["StationId"] == station_id]
        sns.lineplot(x = dataf["Date"].dt.month, y= dataf[indicator], data = dataf, hue = dataf["Date"].dt.year, err_style = None)
    
    plt.show()
plot_pol("All", "PM2.5")
plot_pol("All", "NO")
plot_pol("All", "CO")
plot_pol("All", "SO2")
plot_pol("All", "AQI")
df = station_aqi.loc[station_aqi["StationId"] == "AP001"]
df = df.loc[df["Date"].dt.year != 2020]
df
station2020 = station_aqi[station_aqi["Date"].dt.year == 2020]
station2019 = station_aqi[station_aqi["Date"].dt.year != 2020]
station2020["Month"] = station2020["Date"].dt.month
station2019["Month"] = station2019["Date"].dt.month
station2019["Year"] = 2019
station2020["Year"] = 2020
station2020.groupby("Month").mean()
station2019.groupby("Month").mean()
combdf = pd.concat([station2019.groupby("Month").mean(), station2020.groupby("Month").mean()])
combdf
combdf.reset_index(inplace = True)
sns.lineplot(x = combdf["Month"], y = combdf["PM2.5"], data = combdf, hue = "Year")
