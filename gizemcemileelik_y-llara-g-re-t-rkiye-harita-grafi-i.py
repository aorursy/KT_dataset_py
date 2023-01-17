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
import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
#this library necessary for map

from urllib.request import urlopen

import json

with urlopen('https://raw.githubusercontent.com/cihadturhan/tr-geojson/master/geo/tr-cities-utf8.json') as response:

    harita = json.load(response)

#cities["features"][0]
nufus=pd.read_excel('../input/yllaragrekonut-tr/yllaraveilleregrekonut.xls', index_col=0)

nufus.head()
nufus.index.name = 'Cities'

pop = nufus.reset_index(level='Cities')

pop.tail(10)
new=pop.melt(id_vars=['Cities'], var_name = 'Year', value_name = 'Housing').reset_index(drop=True)

new
id = list(range(1,82))

len(id)

new["id"] = id*5

new
new.dtypes
new['Housing'] = new['Housing'].apply(pd.to_numeric, downcast='float', errors='coerce')

new['Year']=new['Year'].apply(pd.to_numeric, downcast='float', errors='coerce')
new.dtypes
maxi = new.Housing.max()

maxi
mini = new.Housing.min()

mini
new.set_index('id',inplace=True)
import plotly.express as px

enable_plotly_in_cell()



fig = px.choropleth_mapbox(new, geojson=harita, locations=new.index, color=new['Housing'],

                           color_continuous_scale='rainbow',                      

                           hover_name ="Cities",

                           animation_frame=new["Year"],

                           range_color=(mini,maxi),

                           mapbox_style="carto-positron",

                           zoom=5, center={"lat": 38.9597594, "lon": 34.9249653},

                           opacity=0.5,

                           labels={'color':'Housing','ID':'Cities'},

                           title="Total Housing change in Turkey over the years"

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})



fig.show()