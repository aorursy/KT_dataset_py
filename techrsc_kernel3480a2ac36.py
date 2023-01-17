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
import pandas as pd

COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_19_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv")

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")
import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline
import geopandas as gpd

from shapely.geometry import Point, polygon

import descartes
covid_19_data.head()
covid_19_data.dtypes
covid_19_data.isnull().sum()
covid_19_data.describe()
covid_19_data.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].sum()
covid_19_data.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].max()
data_per_data = covid_19_data.groupby('ObservationDate')['Confirmed','Deaths','Recovered'].max()
data_per_data.describe()
data_per_data['Confirmed'].max()
data_per_data.Confirmed.min()
data_per_data.Confirmed.idxmax()
data_per_data.Confirmed.idxmin()
covid_19_data.groupby(['Province/State','Country/Region'])['Confirmed','Deaths','Recovered'].max()
covid_19_data['Country/Region'].value_counts().plot(kind='bar',figsize=(20,10))
len(covid_19_data['Country/Region'].unique())
plt.figure(figsize=(20,10))

covid_19_data['Country/Region'].value_counts().plot.pie(autopct='%1.1f%%')
new_data = gpd.GeoDataFrame(covid_19_data)
new_data.head()
type(new_data)
COVID19_open_line_list.head()
COVID19_open_line_list.groupby(['province','country'])['latitude','longitude'].max()
covid_19
COVID19_open_line_list.columns
COVID19_open_line_list.shape
covid_19_data.shape
covid_19_data.head()
new_data1 = gpd.GeoDataFrame(COVID19_open_line_list)
type(new_data1)
points = [ Point(x,y) for x,y in zip(COVID19_open_line_list.longitude,COVID19_open_line_list.latitude)]
new_data1 = gpd.GeoDataFrame(COVID19_open_line_list,geometry = points)
new_data1.plot(figsize=(20,20))
world_map = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

ax = world_map.plot(figsize=(20,10))

ax.axis('off')
fig, ax = plt.subplots(figsize=(20,10))

new_data1.plot(cmap='Purples',ax=ax)

world_map.geometry.boundary.plot(Color=None,edgecolor='k',linewidth=2,ax=ax)