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
#Visualization

import matplotlib.pyplot as plt

import warnings



warnings.filterwarnings('ignore')

%matplotlib inline
data=pd.read_csv('../input/world-cities-database/worldcities.csv')

data.head()
plt.scatter(data.lng,data.lat)
city=data[['lat','lng']]

city.head()
city_N=city[city.lat>0]      #North  

city_S=city[city.lat<0]      #South



plt.scatter(city_N.lng,city_N.lat)

plt.scatter(city_S.lng,city_S.lat)
city_S.lat=abs(city_S.lat)

plt.scatter(city_S.lng,city_S.lat)
city_SE=city_S[city_S.lng>0] # South East

city_SW=city_S[city_S.lng<0] # South West



city_SE.lng=city_SE.lng-180

city_SW.lng=city_SW.lng+180



# Transfromed map of Southern hemisphere

city_ST=pd.concat([city_SE,city_SW], axis=0)



plt.scatter(city_ST.lng,city_ST.lat)
plt.figure(figsize=(16,10))

plt.title("Antipode Map of the world", fontsize=18)

plt.scatter(city_N.lng,city_N.lat, alpha=0.2)

plt.scatter(city_ST.lng,city_ST.lat, alpha=0.2)