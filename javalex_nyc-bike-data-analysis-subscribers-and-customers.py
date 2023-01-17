# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

import sys



import calendar

import glob

import math



import numpy as np

import pandas as pd

import geopandas

from shapely.geometry import Point, Polygon



import seaborn as sns

import matplotlib.pyplot as plt

%matplotlib inline
YEAR = 2017 #Dataset is from 2017



# Age

AGE_RANGES = ["<20", "20-29", "30-39", "40-49", "50-59", "60+"]

AGE_RANGES_LIMITS = [0, 20, 30, 40, 50, 60, np.inf]

AGE_MIN = 0

AGE_MAX = 100



# Trip duration

DURATION_MIN = 2                 #Assume a minimum duration of 2 seconds

DURATION_MAX = 30 * 24 * 60 * 60 #Assume a maximum duration of 30 days



USERTYPES = ["All", "Subscriber", "Customer"]



# Plotting

FONT_SCALE = 1.5
df = pd.read_csv("/kaggle/input/new-york-city-bike-share-dataset/NYC-BikeShare-2015-2017-combined.csv")

df.describe()
df['Start time'] = pd.to_datetime(df['Start Time'])

df['Stop Time'] = pd.to_datetime(df['Stop Time'])

df['Birth Year'] = pd.to_numeric(df['Birth Year'], downcast='integer')

df.drop(['Unnamed: 0'], axis=1, inplace=True)

df.head()
df["ignore"] = False

df["ignore_reason"] = ""
duplicates = df.duplicated(subset=None, keep='first')

df.insert(len(df.columns), "duplicate", duplicates, allow_duplicates = True)

print("Found {} duplicate rows".format(len(df[duplicates])))
df.loc[df["Bike ID"].isna(), "ignore_reason"] += "Bike ID empty; "

df.loc[df["Start Station ID"].isna(), "ignore_reason"] += "Start Station empty; "

df.loc[df["End Station ID"].isna(), "ignore_reason"] += "End Station empty; "

df.loc[~df["User Type"].isin(["Subscriber", "Customer"]), "ignore_reason"] += "User Type invalid; "
df["age"] = YEAR - df["Birth Year"]

print("Max age: {}.".format(df["age"].max()))

df.loc[df["age"] > AGE_MAX, "ignore_reason"] += "implausible age; "

df.loc[df["age"] < AGE_MIN, "ignore_reason"] += "implausible age; "



print("Min duration: {}.".format(df["Trip Duration"].min()))

df.loc[df["Trip Duration"] < DURATION_MIN, "ignore_reason"] += "Trip Duration implausible; "



print("Max duration: {}.".format(df["Trip Duration"].max()))

df.loc[df["Trip Duration"] > DURATION_MAX, "ignore_reason"] += "Trip Duration implausible; "



df.loc[df["duplicate"] == True, "ignore_reason"] += "duplicate; "
df_subscribers = df[df["User Type"] == "Subscriber"]

df_customers = df[df["User Type"] == "Customer"]

DATAFRAMES = [df, df_subscribers, df_customers]
from numpy.random import seed

seed(42)
for i in range(3):

    dfr = DATAFRAMES[i]

    print(USERTYPES[i])

    print(dfr["Gender"].describe())

    with sns.plotting_context("notebook", font_scale=FONT_SCALE):

        f = sns.countplot(x = "Gender", data=DATAFRAMES[i])

        plt.show()
df["age_range"] = pd.cut(df["age"], AGE_RANGES_LIMITS, labels=AGE_RANGES)
df_subscribers = df[df["User Type"] == "Subscriber"]

df_customers = df[df["User Type"] == "Customer"]

DATAFRAMES = [df, df_subscribers, df_customers]
for i in range(3):

    dfr = DATAFRAMES[i]

    print(USERTYPES[i])

    print(dfr["age"].describe())

    for x in AGE_RANGES:

        print("Age {}: {}".format(x, len(dfr[dfr["age_range"] == x])))

    with sns.plotting_context("notebook", font_scale=FONT_SCALE):

        f = sns.countplot(x = "age_range", data=DATAFRAMES[i])

        f.get_figure().get_axes()[0].set_yscale('log')

        plt.show()
df["station_total"] = df.groupby(["Start Station ID"])["Start Station ID"].transform("count")



max_station_total = df["station_total"].max()

df["station_total_plot"] = 10 + 99 * df["station_total"] / max_station_total #Marker size between 10 and 100

    

df_subscribers = df[df["User Type"] == "Subscriber"]

df_customers = df[df["User Type"] == "Customer"]

DATAFRAMES = [df, df_subscribers, df_customers]
for i in range(3):

    dfr = DATAFRAMES[i]

    print(USERTYPES[i])

    

    df_sample = dfr.sample(10000)

    print(df_sample["station_total"].describe())



    geometry = [Point(xy) for xy in zip(df_sample["Start Station Longitude"], df_sample["Start Station Latitude"])]

    gdf = geopandas.GeoDataFrame(df_sample, geometry=geometry)

    f = gdf.plot(figsize=(12, 8), markersize=df_sample["station_total_plot"])

    plt.show()