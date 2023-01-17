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
floating = pd.read_csv('/kaggle/input/coronavirusdataset/SeoulFloating.csv')
 
floating.head()
le = LabelEncoder()
floating.sex = le.fit_transform(floating.sex)
floating.birth_year = pd.to_numeric(floating.birth_year, errors = 'coerce')
floating.sex = pd.to_numeric(floating.sex, errors = 'coerce')
floating.hour = pd.to_numeric(floating.hour, errors = 'coerce')
sns.distplot(floating.fp_num)
sns.violinplot('sex','fp_num', data = floating )
city = floating.groupby(['city'], as_index = False).sum()
city
plt.figure(figsize = (8, 6))
sns.barplot('fp_num','city', data = city)
date = floating.groupby('date', as_index = False).sum()
date
# On any given day the population is fairly constant

plt.figure(figsize = (20,20))
sns.scatterplot('fp_num','date', data = date)
hour = floating.groupby('hour', as_index = False).sum()
hour.head()
# So in a given day this is the distribution of floating population in the cities

sns.lineplot('hour', 'fp_num', data = hour)
birth_year = floating.groupby('birth_year', as_index = False).sum()
birth_year
sns.lineplot('birth_year', 'fp_num', data = birth_year)
