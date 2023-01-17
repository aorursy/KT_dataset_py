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

#Importing plotting tool
import matplotlib.pyplot as plt
df_train = pd.read_csv("../input/ashrae-energy-prediction/train.csv")
df_train.head(5)

print("Number of unique buildings: ", len(df_train.drop_duplicates(subset = "building_id")))
print("Amount of rows with NaN values: ", len(df_train))
print("Amount of rows without NaN values: ", len(df_train.dropna()))
#Bar chart for the overall amount of meter readings for each building ID

df_train["building_id"].value_counts().head()
xtick_values = []

df_train["building_id"].value_counts().plot.bar(figsize = (15,6), xlabel = "Building ID", ylabel = "Number of meter readings", fontsize = 10, xticks = [0], title = "Number of meter readings for each building ID")


#Alternative plt box chart that I cannot get to work, so I just saved it here for now
#plt.bar(df_train["building_id"].value_counts(), height = 50000)
#plt.show()
df_train["meter"].value_counts().plot.bar( xlabel = "amount of readings", ylabel = "meter type", title = "Amount of readings for each meter type")
#plt.bar((df_train["meter"].value_counts()), height = 10000)

df_train.drop(["building_id", "timestamp"], axis = 1).groupby(["meter"]).mean().plot.bar(xlabel = "mean value of the meter readings", title = "Mean meater reading for each meter type")
xtickx = [0]

#for i in range(len(df_train)):
    #xtickx += [df_train["timestamp"][i]]
df_train.filter(["meter_reading", "timestamp"]).plot(figsize = (20,7), title = "all meter readings for each building id" )
#Creating one dataframe for each meter type

df_train_0 = df_train.loc[df_train["meter"] == 0]
df_train_1 = df_train.loc[df_train["meter"] == 1]
df_train_2 = df_train.loc[df_train["meter"] == 2]
df_train_3 = df_train.loc[df_train["meter"] == 3]

#Calculating the mean readings of the buildings
#In this case, it can be seen that the building_id will not make any sense anymore, 
#since it calculates the mean of the IDs. It doesnt matter since the mean meter_reading is of interest

df_train_0.groupby(by = "timestamp").mean().filter(["timestamp", "meter_reading"]).plot(figsize =(15,7), ylabel = "mean meter readings", title = "mean hourly electricity readings (index 0)")
df_train_1.groupby(by = "timestamp").mean().filter(["timestamp", "meter_reading"]).plot(figsize =(15,7), ylabel = "mean meter readings", title = "mean hourly chilled water readings (index 1)")
df_train_2.groupby(by = "timestamp").mean().filter(["timestamp", "meter_reading"]).plot(figsize =(15,7), ylabel = "mean meter readings", title = "mean daily steam readings (index 2)")
df_train_3.groupby(by = "timestamp").mean().filter(["timestamp", "meter_reading"]).plot(figsize =(15,7), ylabel = "mean meter readings", title = "mean hourly hot water readings (index 3)")
#print("Amount of sites: ",len(df_building_metadata.drop_duplicates(subset = "site_id")))
df_building_metadata = pd.read_csv("../input/ashrae-energy-prediction/building_metadata.csv")
df_building_metadata.head()

df_building_metadata["site_id"].value_counts(sort = False).plot.bar(figsize = (15, 7), xlabel = "Site IDs", ylabel = "Amount of building IDs", fontsize = 10, title = "Number of buildings on each site")
df_building_metadata["primary_use"].value_counts().plot.bar(figsize = (14,5), xlabel = "Primary use", ylabel = "Number of buildings", fontsize = 10, rot = 90, title = "Number of buildings used for a particual reason")
#print(df_building_metadata.set_index(keys=["site_id", "building_id"]))
#print(df_building_metadata.filter(["site_id", "primary_use"]))

df_building_metadata.filter(["site_id", "primary_use"]).value_counts().sort_values().reset_index().pivot(index = "site_id", columns = "primary_use", values = 0).plot.bar(stacked = True,figsize = (20,10), xlabel = "Site ID", ylabel = "Number of buildings", fontsize = 10, rot = 90, title = "Number of buildings used for a particual reason" )
#
df_weather = pd.read_csv("../input/ashrae-energy-prediction/weather_train.csv")
df_weather.head()
df_weather.filter(["timestamp", "air_temperature"]).groupby(by = "timestamp").mean().plot(figsize = (15,7), xlabel = "mean air temperature", title = "Mean air temperature for all sites")

print("Correlation hot water meter and air temperature: ",df_weather["air_temperature"].corr(df_train_3["meter_reading"]))
print("Correlation steam meter and air temperature: ",df_weather["air_temperature"].corr(df_train_2["meter_reading"]))
print("Correlation chilled water meter and air temperature: ",df_weather["air_temperature"].corr(df_train_1["meter_reading"]))
print("Correlation electricity meter and air temperature: ",df_weather["air_temperature"].corr(df_train_0["meter_reading"]))