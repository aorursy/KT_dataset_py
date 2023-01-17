# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
path = "../input"
os.chdir(path)
data = pd.read_csv("../input/crime.csv", encoding = "ISO-8859-1", low_memory=False)
data.head()
data.tail()
data.columns
data.DISTRICT.value_counts()
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(111)
sns.countplot("DISTRICT", data = data, ax = ax)
plt.show()
plt.figure(figsize=(16,8))
data['DISTRICT'].value_counts().plot.bar()
sns.catplot(x="DISTRICT",       # Variable whose distribution (count) is of interest

            hue="MONTH",      # Show distribution, pos or -ve split-wise

            col="YEAR",       # Create two-charts/facets, gender-wise

            data=data,

            kind="count")
plt.figure(figsize=(16,8))
data['DISTRICT'].loc[data['YEAR']==2015].value_counts().plot.bar()
plt.figure(figsize=(16,8))
data['OFFENSE_CODE_GROUP'].value_counts().plot.bar()
plt.show()
plt.figure(figsize=(16,8))
sns.countplot(x='YEAR', data = data)
plt.figure(figsize=(16,8))
top10cloc = data.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=False)
top10cloc = top10cloc [:10]
top10cloc.plot(kind='bar', color='green')
bot10cloc = data.groupby('DISTRICT')['INCIDENT_NUMBER'].count().sort_values(ascending=True)
bot10cloc = bot10cloc [:10]
bot10cloc.plot(kind='bar', color='blue')
plt.figure(figsize=(15,7))
data.groupby(['DAY_OF_WEEK'])['INCIDENT_NUMBER'].count().plot(kind = 'bar')
plt.figure(figsize=(17,9))
data.groupby(['DISTRICT'])['STREET'].count().plot(kind = 'bar')
groups = data['DISTRICT'].unique()
n_groups = len(data['DISTRICT'].unique())-1
index = np.arange(n_groups)
bar_width = 0.2
opacity= 0.8
plt.figure(figsize=(16,8))
dy = data[['DISTRICT','YEAR']]



dy_2015 = dy.loc[(dy['YEAR'] == 2015)]

dy_2016 = dy.loc[(dy['YEAR'] == 2016)]

dy_2017 = dy.loc[(dy['YEAR'] == 2017)]

dy_2018 = dy.loc[(dy['YEAR'] == 2018)]



cri_2015 = dy_2015['DISTRICT'].value_counts()

cri_2016 = dy_2016['DISTRICT'].value_counts()

cri_2017 = dy_2017['DISTRICT'].value_counts()

cri_2018 = dy_2018['DISTRICT'].value_counts()



bar1 = plt.bar(index, cri_2015, bar_width, alpha = opacity, color = 'r', label = '2015')

bar2 = plt.bar(index + bar_width, cri_2016, bar_width, alpha = opacity, color = 'g', label = '2016')

bar3 = plt.bar(index+ bar_width+ bar_width, cri_2017, bar_width, alpha = opacity, color = 'b', label = '2017')

bar4 = plt.bar(index+ bar_width+ bar_width+ bar_width, cri_2018, bar_width, alpha = opacity, color = 'y', label = '2018')
from mpl_toolkits.basemap import Basemap, cm
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon
%matplotlib inline
fig,ax = plt.subplots(figsize = (15,15))
geometry = [Point(xy) for xy in zip( data["Long"],data["Lat"])]
geometry[:3]
geo_data = gpd.GeoDataFrame(data, geometry = geometry)
geo_data.head()
fig,ax = plt.subplots(figsize = (15,15))
from mpl_toolkits.basemap import Basemap
import folium
from folium import plugins
import statsmodels.api as sm
data[['Lat','Long']].describe()
location = data[['Lat','Long']]
location = location.dropna()
location = location.loc[(location['Lat']>40) & (location['Long'] < -60)]
x = location['Long']
y = location['Lat']
colors = np.random.rand(len(x))
plt.figure(figsize=(20,20))
plt.scatter(x, y,c=colors, alpha=0.5)
plt.show()
m = folium.Map([42.348624, -71.062492], zoom_start=11)
m
x = location['Long']
y = location['Lat']
sns.jointplot(x, y, kind='scatter')
sns.jointplot(x, y, kind='hex')
sns.jointplot(x, y, kind='kde')