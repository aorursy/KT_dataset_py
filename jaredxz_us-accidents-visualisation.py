# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

from datetime import datetime

import geopandas as gpd # for maps



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
# STILL WORK IN PROGRESS, BUT AM TRYING TO PRACTICE MY PYTHON SKILLS! COMMENTS TO IMPROVE ARE WELCOMED!



raw_df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv", parse_dates=["Start_Time", "End_Time"])

raw_df.head()
# GET ONLY TIME OF ACCIDENT

raw_df["START_TM"] = [x.replace(year = 2000, month = 1, day = 1) for x in raw_df["Start_Time"]]

raw_df.columns
# fig,ax = plt.subplots()

# ax.hist(raw_df["Visibility(mi)"], bins = 20)

# plt.show()

# raw_df["Visibility(mi)"].value_counts().sort_index()

# raw_df["Visibility(mi)"].describe()
fig, ax = plt.subplots(figsize = (18,5))

locator = mdates.HourLocator()

formatter = mdates.DateFormatter("%H:%M")



ax.hist(raw_df["START_TM"], bins = 24, edgecolor = "whitesmoke")



ax.xaxis.set_major_locator(locator)

ax.set_xlim(datetime(2000,1,1,0,0,0), datetime(2000,1,2,0,0,0))

xticks = ax.get_xticks()

ax.set_xticklabels(xticks, rotation = 90)

ax.xaxis.set_major_formatter(formatter)

ax.set_title("Histogram of Accident Start Times")

plt.show()
raw_df["START_DT"] = [x.replace(hour = 0,minute = 0, second=0) for x in raw_df["Start_Time"]]

raw_df["START_DT"].value_counts().sort_index()
# TIMESERIES OF ACCIDENTS

fig, ax = plt.subplots(figsize = (18,5))



temp = raw_df["START_DT"].value_counts().sort_index()



locator = mdates.MonthLocator()

formatter = mdates.DateFormatter("%Y-%m")

ax.plot(temp.index, temp)



ax.set_xlim(min(temp.index), max(temp.index))

ax.xaxis.set_major_locator(locator)

ax.set_xticklabels(temp.index, rotation = 90)

ax.xaxis.set_major_formatter(formatter)



plt.show()
# TOP 50 CITIES WITH HIGHEST ACCIDENTS

city_df = raw_df["City"].value_counts().head(50)



fig,ax = plt.subplots(figsize = (18,5))

ax.bar(x = city_df.index, height=city_df)

ax.set_xticklabels(city_df.index,rotation = 90)

plt.show()
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

usa_df = world[world["name"] == "United States of America"]

ax1 = usa_df.plot(color = "whitesmoke", edgecolor = "black", linestyle = ":", figsize = (18,18))



# for sev_level in sorted(raw_df["Severity"].unique()):

sev_level = 4

temp_df = raw_df[raw_df["Severity"]==sev_level]

accidents_location = gpd.GeoDataFrame(temp_df, geometry = gpd.points_from_xy(temp_df["Start_Lng"], temp_df["Start_Lat"]))

accidents_location.plot(markersize = 2, ax= ax1,  facecolors='none', edgecolors='r', alpha = 0.5)



plt.show()