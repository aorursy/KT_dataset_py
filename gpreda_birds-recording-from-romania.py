import pandas as pd

import numpy as np

from pandas_profiling import ProfileReport

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline
data_df = pd.read_csv("/kaggle/input/xenocanto-birds-from-romania/birds_romania.csv")

profile = ProfileReport(data_df, title="Pandas Profiling Report")
profile
aggregated_df = data_df.groupby(["lat", "lng"])["id"].count().reset_index()

aggregated_df.columns = ['lat', 'lng', 'count']

m = folium.Map(location=[46, 26], zoom_start=7)

max_val = max(aggregated_df['count'])

HeatMap(data=aggregated_df[['lat', 'lng', 'count']],\

        radius=30, max_zoom=18).add_to(m)

m