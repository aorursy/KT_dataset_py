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
!pip install calmap

import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import folium
from folium.plugins import HeatMapWithTime, TimestampedGeoJson
import matplotlib.style as style 
style.use('fivethirtyeight')
import numpy as np; np.random.seed(sum(map(ord, 'calmap')))
import pandas as pd
import calmap
# Pandas to read Covid Tweets dataset
tweets = pd.read_csv('../input/covid19-tweets/covid19_tweets.csv')
# Examining the dataset from begining
tweets.head()
# Checking the shape of dataset
tweets.shape
tweets.info()
# World City Dataset

cities = pd.read_csv('../input/world-cities-datasets/worldcities.csv')
# Exploring city dataset
cities.head()
## Duplicate Location in Tweets Dataset

tweets["location"] = tweets["user_location"]
tweets["country"] = np.NaN
user_location = tweets['location'].fillna(value='').str.split(',')
lat = cities['lat'].fillna(value = '').values.tolist()
lng = cities['lng'].fillna(value = '').values.tolist()
country = cities['country'].fillna(value = '').values.tolist()

# Getting all alpha 3 codes into  a list
world_city_iso3 = []
for c in cities['iso3'].str.lower().str.strip().values.tolist():
    if c not in world_city_iso3:
        world_city_iso3.append(c)
        
# Getting all alpha 2 codes into  a list    
world_city_iso2 = []
for c in cities['iso2'].str.lower().str.strip().values.tolist():
    if c not in world_city_iso2:
        world_city_iso2.append(c)
        
# Getting all countries into  a list        
world_city_country = []
for c in cities['country'].str.lower().str.strip().values.tolist():
    if c not in world_city_country:
        world_city_country.append(c)

# Getting all amdin names into  a list
world_states = []
for c in cities['admin_name'].str.lower().str.strip().tolist():
    world_states.append(c)


# Getting all cities into  a list
world_city = cities['city'].fillna(value = '').str.lower().str.strip().values.tolist()



for each_loc in range(len(user_location)):
    ind = each_loc
    each_loc = user_location[each_loc]
    for each in each_loc:
        each = each.lower().strip()
        if each in world_city:
            order = world_city.index(each)
            tweets['country'][ind] = country[order]
            continue
        if each in world_states:
            order= world_states.index(each)
            tweets['country'][ind] = country[order]
            continue
        if each in world_city_country:
            order = world_city_country.index(each)
            tweets['country'][ind] = world_city_country[order]
            continue
        if each in world_city_iso2:
            order = world_city_iso2.index(each)
            tweets['country'][ind] = world_city_country[order]
            continue
        if each in world_city_iso3:
            order = world_city_iso3.index(each)
            tweets['country'][ind] = world_city_country[order]
            continue

print('Total Number of valid Tweets Available: ',tweets['country'].isnull().sum())
tweet_per_country = tweets['country'].str.lower().dropna()
tw = tweet_per_country.value_counts().rename_axis('Country').reset_index(name='Tweet Count')
print(tw)
plt.rcParams['figure.figsize'] = (15,10)
plt.title('Top 20 Countries with Most Tweets',fontsize=15)
sns.set_palette("husl")
ax = sns.barplot(y=tw['Country'].head(20),x=tw['Tweet Count'].head(20))