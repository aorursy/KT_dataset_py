# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
import folium
from folium.plugins import HeatMap
import matplotlib as plt
df = pd.read_csv("/kaggle/input/us-accidents/US_Accidents_Dec19.csv")
df.head()
df = df[["Severity", "Start_Lat", "Start_Lng"]]
df.shape

df3 = df[df['Severity'] > 3]  

df3.shape
df3.head()

severities = df[['Severity']]
severities.head()
df3 = df3[["Start_Lat", "Start_Lng"]]
map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 13)

HeatMap(df3).add_to(map_van)
map_van
df.head()
df4 = df[df["Severity"]>3]
df4.shape
df4.head()
df4 = df4[["Start_Lat", "Start_Lng"]]
map_van = folium.Map(location = [37.773972, -122.431297], zoom_start = 13)

HeatMap(df4).add_to(map_van)
map_van
