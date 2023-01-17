#set input here

lat = 42.65
#Kaggle default cell





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
#pull in city data and take a look

cities_data = pd.read_csv("/kaggle/input/world-cities-database/worldcitiespop.csv")

cities_data.head(5)
#filter to only cities with greater population > 100K

big_cities = cities_data.loc[cities_data['Population'] > 100000]

#boundaries

lat_min = lat - 0.5

lat_max = lat + 0.5
#create new dataframe of only cities within boundary

cities_near = big_cities.loc[(big_cities['Latitude'] >= lat_min) & (big_cities['Latitude'] <= lat_max)]
#create column of distance from input latitude incase want to sort by this

cities_near['Delta'] = abs(cities_near['Latitude'] - lat)
#sort and display

cities_near = cities_near.sort_values(by=['Population'],ascending=[False])

cities_near.head(30)
#if you want to save the output of big_cities to play with

#or change to cities_near

#big_cities.to_csv('/kaggle/working/big_cities.csv')