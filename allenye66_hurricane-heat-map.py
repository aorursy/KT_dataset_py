import pandas as pd
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap
import os
os.listdir("/kaggle/input/hurricane-database")

atlanticDF = pd.read_csv("/kaggle/input/hurricane-database/atlantic.csv")
atlanticDF = atlanticDF[["Latitude", "Longitude"]]
pacificDF = pd.read_csv("/kaggle/input/hurricane-database/pacific.csv")
pacificDF = pacificDF[["Latitude", "Longitude"]]
pacificDF.head()
atlanticDF.head()
map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 13)

pacificDF = pacificDF.replace({'N':''}, regex=True)
pacificDF = pacificDF.replace({'W':''}, regex=True)
pacificDF = pacificDF.replace({'E':''}, regex=True)
pacificDF = pacificDF.replace({'S':''}, regex=True)

atlanticDF = atlanticDF.replace({'N':''}, regex=True)
atlanticDF = atlanticDF.replace({'W':''}, regex=True)
atlanticDF = atlanticDF.replace({'E':''}, regex=True)
atlanticDF = atlanticDF.replace({'S':''}, regex=True)

pacificDF.head()
atlanticDF.head()
HeatMap(atlanticDF).add_to(map_van)
HeatMap(pacificDF).add_to(map_van)
map_van