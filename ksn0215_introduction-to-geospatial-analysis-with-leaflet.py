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
import folium

from IPython.display import HTML

from folium.plugins import HeatMap
df = pd.read_csv("../input/traffic-collision-data-from-2010-to-present.csv")
df.head()
df['Location'].head()
longitude = df['Location'].str.extract('\'longitude\':\s\'(.+)\'}', expand=True)

longitude = longitude.applymap(float)

latitude = df['Location'].str.extract('{\'latitude\':\s\'(.+)\',\s\'human_address\'', expand=True)

latitude = latitude.applymap(float)

longitude = longitude.rename(columns={0: 'lon'})

latitude = latitude.rename(columns={0: 'lat'})

coordinate = pd.concat([latitude, longitude], axis=1)
coordinate.head()
df_new = pd.concat([df, coordinate], axis=1)
df_new.head(3)
map_la = folium.Map(location=(34.052235,-118.243683))

data = []

for i in range(len(df_new)):

    data.append((df_new['lat'][i],df_new['lon'][i]))

HeatMap(data,radius=9).add_to(map_la)

map_la.save('la_accidents.html')
HTML(r'<iframe width="800" height="500" frameborder="0" allowfullscreen src="./la_accidents.html"></iframe>')