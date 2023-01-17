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
for_map = pd.read_csv("../input/911.csv")
for_map=for_map.groupby(['lat','lng'])['lat'].count()
for_map=for_map.to_frame()
for_map.columns.values[0]='count1'
for_map=for_map.reset_index()
for_map
import pandas as pd 
import folium
from folium.plugins import HeatMap

max_amount = float(for_map['count1'].max())
lats=for_map[['lat','lng','count1']].values.tolist()

lats

hmap = folium.Map(location=[40.5, -75.5], zoom_start=7, )
hmap.add_child(HeatMap(lats, radius = 5))
folium.Marker([40.2, -75.5],popup='emergency points in th cities ' ).add_to(hmap)

hmap