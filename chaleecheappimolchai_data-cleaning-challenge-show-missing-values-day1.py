# modules we'll use

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns







df_summary = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Measurement_summary.csv")

df_summary.head()
df_meas_info = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_info.csv")

df_meas_info.head()
df_meas_item = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_item_info.csv")

df_meas_item.head()
df_meas_station = pd.read_csv("../input/air-pollution-in-seoul/AirPollutionSeoul/Original Data/Measurement_station_info.csv")

df_meas_station.head()
# get the number of missing data points per column

missing_values_count = df_summary.isnull().sum()



# look at the # of missing points in the first ten columns

missing_values_count
# get the number of missing data points per column

missing_df_info = df_meas_info.isnull().sum()

missing_df_info
# get the number of missing data points per column

missing_df_item = df_meas_item.isnull().sum()

missing_df_item 

# get the number of missing data points per column

missing_df_station = df_meas_station.isnull().sum()

missing_df_station