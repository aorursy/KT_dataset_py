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
## importing folium and using FL as an alias (because i hate writting)



import folium as fl
## using method chaining to select the data we need and dropna



df = (

    

    pd.read_csv('/kaggle/input/denver-small-cell-nodes/small_cell_nodes.csv')

    .loc[:,['OWNERNAME','NODE_N','NODE_E','POLEHEIGHT_FT']]

    .dropna()



)



df.head()
## Creating a folium map with standard markers





m = fl.Map(location=[39.742793, -105.0109598],zoom_start=11)



for lat,long,owner in zip(df.NODE_N,df.NODE_E,df.OWNERNAME):

    fl.Marker([lat,long],popup=owner).add_to(m)



m
## Bring the heat!! Let's build a heatmap using folium and setting a small radius for each tower



from folium.plugins import HeatMap



heat = fl.Map(location=[39.742793, -105.0109598],zoo_start=11)



data = [[lat,long] for lat,long in zip(df.NODE_N,df.NODE_E)]



HeatMap(data,radius=15).add_to(heat)



heat