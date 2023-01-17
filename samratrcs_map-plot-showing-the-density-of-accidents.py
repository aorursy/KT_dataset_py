# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium
from folium.plugins import HeatMap

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

#Plot the density of the emergencies on MAP
accdf = pd.read_csv('../input/911.csv')
accdf=accdf.groupby(['lat','lng'])['lat'].count()
accdf=accdf.to_frame()
accdf.columns.values[0]='count1'
accdf=accdf.reset_index()
lats=accdf[['lat','lng','count1']].values.tolist()
    
hmap = folium.Map(location=[40.4, -75.2], zoom_start=9, )
hmap.add_child(HeatMap(lats, radius = 5))
hmap

# Any results you write to the current directory are saved as output.
