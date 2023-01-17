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
ff=pd.read_csv("../input/FastFoodRestaurants.csv")
ff.head()
print(ff.describe())
print(ff.dtypes)
# only one country in the dataset
ff['country'].unique()
# irrelevant column: delete
del ff['country']
# number of cities covered: 2775
cities = ff['city'].unique().tolist()
len(cities)
# for each of the cities, how many restaurants do we have?
ff_per_city = ff.groupby(['city'])['name'].count().to_frame()
ff_per_city.plot()
# how many restaurant chains do we have?: 548
chains_per_city = ff.groupby(['name']).count().index.tolist()
len(chains_per_city)
# Where am I?
# Locating myself using a simple IP lookup
# Website: ip.info.io/json
from urllib import request
# request provides the functionality to open urls
import json
# use json to convert the web response into a Python dictionary
# errors can occur when accessing web resources
try:
    data = json.load(request.urlopen('http://ipinfo.io/json'))
except Exception as e:
    print(e)
else:
    print('Your location is as follows: {city},{region},{country}'.format(**data))
# Latitude and Longitude information
lat = data['loc'].split(',')
lat = [float(x) for x in lat]
lati = lat[0]
longi = lat[1]
lati = 34.0657
longi = -118.436
# How many restaurants are close to me?
type(ff['latitude'][0])
type(ff['longitude'][0])


# Positive latitude is above the equator (N), and negative latitude is below the equator (S)
# Positive longitude is east of the prime meridian, 
# while negative longitude is west of the prime meridian (a north-south line that runs through a point in England)


ff['Sum of diff distances of squared'] = ((lati - ff['latitude'])**2) + ((longi - ff['longitude'])**2)
ff_for_me = ff.sort_values(by=['Sum of diff distances of squared'])
ff_for_me.head()