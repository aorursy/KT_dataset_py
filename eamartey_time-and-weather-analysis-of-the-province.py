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
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import re
import datetime
import folium
import geopandas as gpd
from shapely.geometry import Point, Polygon
from geopandas import GeoDataFrame
import math
from folium.plugins import HeatMap, MarkerCluster
from folium import Marker,GeoJson,Choropleth, Circle
time_prov = pd.read_csv('/kaggle/input/coronavirusdataset/TimeProvince.csv')
weather = pd.read_csv('/kaggle/input/coronavirusdataset/Weather.csv')
time_prov.head()
time = time_prov.groupby('date', as_index = False).sum()
time
sns.scatterplot('released', 'confirmed', data = time)
sns.scatterplot('released', 'confirmed', data = time, hue = 'deceased')

sns.scatterplot('deceased', 'confirmed', data = time)
 
sns.scatterplot('deceased', 'released', data = time)
time_prov.head()
# Daegu and certain provinces have high confirmed values

sns.barplot('confirmed','province', data = time_prov) 
# They also have high released values

sns.barplot('released','province',  data = time_prov) 
# Gyeongsangbuk-do is the only province with disproportionally high deceased number of deaths. 
# Seoul, though, has lower deaths than expected. The question for both is why?

sns.barplot('deceased','province', data = time_prov)
sns.scatterplot('time', 'confirmed', data = time_prov)
weather.head()
# Grouping...

date = weather.groupby('date', as_index = False).sum() 
date
# The average temperature increases the average relative humidity  

sns.scatterplot('avg_temp', 'avg_relative_humidity', data = date)
# The most wind direction decreases the average temperature
sns.scatterplot('avg_temp', 'most_wind_direction', data = date)
# The maximum wind speed decreases the average temperature slightly
sns.scatterplot('avg_temp', 'max_wind_speed', data = date)
# This makes sense that it's a one-to-one correlation

sns.scatterplot('avg_temp', 'min_temp', data = date) 
# Same thing here

sns.scatterplot('avg_temp', 'max_temp', data = date) 
# Precipitation has no bearing on the average temperature

sns.scatterplot('avg_temp', 'precipitation', data = date)

province = weather.groupby(['province'], as_index = False).sum()
province
sns.barplot('avg_temp', 'province', data = province)
sns.barplot('max_wind_speed', 'province', data = province)
sns.barplot('precipitation', 'province', data = province)
sns.barplot('most_wind_direction', 'province', data = province)
sns.barplot('avg_relative_humidity', 'province', data = province)
