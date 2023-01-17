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
import pandas as pd

import plotly.express as px

# loading Turkey's geoplot json file

from urllib.request import urlopen

import json

with open("../input/geoplot/tr-cities-utf8.json") as f:

    cities = json.load(f)
import pandas as pd

df = pd.read_csv('../input/trpopulation/TRNufus.csv')

df.set_index('Id', inplace=True)

df.head()
import plotly.express as px



fig = px.choropleth_mapbox(df, geojson=cities, locations=df.index, color=np.log10(df["Pop"]),hover_name="City", animation_frame=df["Year"],

                           color_continuous_scale='twilight',

                           

                           mapbox_style="carto-positron",

                           zoom=4, center = {"lat": 38.963745, "lon": 35.243322},

                           opacity=0.7,

                           labels={'color':'Population','Id': 'City'}

                          )



fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()