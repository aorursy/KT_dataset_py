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
print("Hello World, I am CORONA :)")
import folium
import pandas as pd
import os
import json
dataset_name="vignesh1694/covid19-coronavirus"
filename="/kaggle/input/covid19-coronavirus/time_series_covid19_confirmed.xlsx"

df = pd.read_excel(filename)
!pwd
!ls
df.head()
# transform your dataset to coalesce the Province/State and the Country/Region
df['name']=df['Province/State'].mask(pd.isnull, df['Country/Region'])
df.head()
# create an empty map
map = folium.Map(zoom_start=1.5,width=1000,height=750,location=[0,0], tiles = 'Stamen Toner')
# loop on your date to populate the map
for row in df.itertuples():
    lat=getattr(row, "Lat")
    long=getattr(row, "Long")
    confirmed=int(row[-2])
    name=getattr(row, "name")
    tooltip = f"{name} - {confirmed}"
    radius = 30 if confirmed/10>30 else confirmed/10

    if confirmed>0:
        folium.vector_layers.CircleMarker(
            location=(lat, long),
            radius=radius,
            tooltip=tooltip,
            color='red',
            fill_color='red'
        ).add_to(map)
map