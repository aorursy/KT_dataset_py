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
#importing necessery libraries for future analysis of the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
# Pandas to read NYC Airbnb dataset
nyc_airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# Examining the dataset from begining
nyc_airbnb.head()
# Examining the dataset from the end
nyc_airbnb.tail()
# Checking the shape of dataset
nyc_airbnb.shape
# Checking the datatype and non-null listings of each parameter in the datset
nyc_airbnb.info()
# Checking the data distribution of necessary paramters
nyc_airbnb[['price','minimum_nights','availability_365']].describe()
# Checking if the data contains any NULL value
nyc_airbnb.isnull().sum()
nyc_airbnb[['last_review','reviews_per_month']]
nyc_airbnb.fillna({'reviews_per_month':0}, inplace=True)
# lets see about last reviews
nyc_airbnb['last_review'].dropna()
year = nyc_airbnb['last_review'].str.split('-').str.get(0).dropna().to_frame()
print(year['last_review'].value_counts())
plt.rcParams['figure.figsize'] = (12,6)
plt.title('Last Review vs Year',fontsize=15)
sns.set(style="darkgrid")
sns.set_palette("husl")
ax = sns.countplot(x=year['last_review'])
nyc_airbnb['neighbourhood_group'].value_counts()
neighbourhood_groups = nyc_airbnb['neighbourhood_group']
sns.set(style="darkgrid")
colors = ['orange', 'pink', 'crimson', 'lightgreen', 'black']
plt.title('Neighbourhood Group Density',fontsize=15)
circle = plt.Circle((0, 0), 0.6, color = 'white')
neighbourhood_groups.value_counts().plot(kind='pie', figsize=(8, 8), rot=1, colors=colors)
p = plt.gcf()
p.gca().add_artist(circle)
plt.axis('off')
plt.legend()

plt.show()
sns.set(style="darkgrid")
colors = ['orange', 'pink', 'crimson', 'lightgreen', 'black']
plt.title('Room Type in Newyork',fontsize=15)
sns.set_palette("husl")
ax = sns.countplot(x=nyc_airbnb['room_type'])
plt.title('Price Distribution',fontsize=15)
sns.kdeplot(nyc_airbnb['price'], shade='True', legend='True')
plt.title('Minimum Nights',fontsize=15)
sns.stripplot(nyc_airbnb['minimum_nights'], palette='BuGn_r')
plt.title('Availability of Rooms throughout the year',fontsize=15)
sns.boxplot(nyc_airbnb['availability_365'], palette='Blues_d')
data=nyc_airbnb[(nyc_airbnb.reviews_per_month>0)]
data.sort_values(by=['reviews_per_month'],inplace=True, ascending=False)
sns.set(style="darkgrid")
sns.set_palette("muted")
a = sns.catplot(x="neighbourhood_group", y="reviews_per_month",kind="swarm", data=data.head(1000),height=7)
#using violinplot to showcase density and distribtuion of prices 
sns.set_palette("husl")
viz_2=sns.violinplot(data=nyc_airbnb[(nyc_airbnb.price<500) & (nyc_airbnb.room_type != 'Shared room')], x='neighbourhood_group', y='price', hue='room_type', height=5,split=True)
#viz_2=sns.boxenplot(data=nyc_airbnb[nyc_airbnb.price<500], x='neighbourhood_group', y='price')
viz_2.set_title('Density and distribution of prices for each neighberhood_group')
sns.set_palette("RdGy")
viz_2=sns.boxenplot(data=nyc_airbnb[(nyc_airbnb.minimum_nights<50) & (nyc_airbnb.room_type != 'Shared room')], x='neighbourhood_group', y='minimum_nights',hue='room_type')
viz_2.set_title('Density and distribution of prices for each neighberhood_group')
viz_2=sns.scatterplot(data=nyc_airbnb, x='latitude', y='longitude',hue='availability_365')
viz_2.set_title('Density and distribution of prices for each neighberhood_group')
import pandas as pd 
import folium
from folium.plugins import HeatMap

max_amount = float(nyc_airbnb['price'].max())

hmap = folium.Map(location=[40.73, -73.95], zoom_start=11, )

hm_wide = HeatMap( list(zip(nyc_airbnb.latitude.values, nyc_airbnb.longitude.values, nyc_airbnb.price.astype('float64').values)),
                   min_opacity=0.1,
                   max_val=max_amount,
                   radius=10, blur=15, 
                   max_zoom=5, 
                 )

hmap.add_child(hm_wide)

# correlation matrix
sns.set(font_scale=3)
plt.figure(figsize=(30, 20))
sns.heatmap(nyc_airbnb.corr(), annot=True)