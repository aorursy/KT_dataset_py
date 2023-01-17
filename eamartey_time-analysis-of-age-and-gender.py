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
time_age = pd.read_csv('/kaggle/input/coronavirusdataset/TimeAge.csv')
time_gender = pd.read_csv('/kaggle/input/coronavirusdataset/TimeGender.csv')

time_age.head()
time_age['age'] =  time_age['age'].str.rstrip('s')
time_age.head()
# Renaming the 'age' column

time_age.rename(columns = {'age':'age_group'}, inplace = True)
time_age.head()
sns.boxplot('age_group', 'confirmed', data = time_age) 

sns.lineplot('age_group', 'deceased', data = time_age)  
sns.scatterplot('confirmed', 'deceased', data = time_age, hue = 'age_group')

time_age.date.value_counts()
time = time_age.groupby('date', as_index = True).sum()
time 
sns.lineplot('confirmed', 'deceased', data = time) # The number of deaths was definitely reaching a plateau
time_age.time.value_counts() 
time_gender.head() 
time_gender.time.value_counts()
sns.violinplot('sex', 'confirmed', data = time_gender)  
sns.violinplot('sex', 'deceased', data = time_gender)  
sns.scatterplot('deceased', 'confirmed', data = time_gender, hue = 'sex')
 
plt.figure(figsize = (20, 18))
sns.scatterplot('deceased', 'date', data = time_gender, hue = 'sex')
 
gender = time_gender.groupby('date').sum()
gender
sns.scatterplot('deceased', 'confirmed', data = gender)
 

