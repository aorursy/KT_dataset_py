import pandas as pd

import numpy as np

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

import folium

from pandas_profiling import ProfileReport

from folium.plugins import HeatMap, HeatMapWithTime

%matplotlib inline
data_df = pd.read_csv("/kaggle/input/xenocanto-birds-from-france/birds_france.csv")
profile = ProfileReport(data_df, title="Pandas Profiling Report")
aggregated_df = data_df.groupby(["lat", "lng"])["id"].count().reset_index()

aggregated_df.columns = ['lat', 'lng', 'count']

m = folium.Map(location=[47, 0], zoom_start=6)

max_val = max(aggregated_df['count'])

HeatMap(data=aggregated_df[['lat', 'lng', 'count']],\

        radius=15, max_zoom=12).add_to(m)

m