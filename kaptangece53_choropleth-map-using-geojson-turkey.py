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
from urllib.request import urlopen

import json

with open('/kaggle/input/countries/counties2.json', 'r',encoding="utf-8-sig") as f:

    distros_dict = json.load(f)

import pandas as pd

df = pd.read_csv("/kaggle/input/fipsunempsample/fips-unemp-sample.csv",

                   dtype={"fips": str})



import plotly.express as px



fig = px.choropleth_mapbox(df, geojson=distros_dict, locations='fips', color='unemp',

                           color_continuous_scale="Viridis",

                           range_color=(0, 12),

                           mapbox_style="carto-positron",

                           zoom=5, center={"lat": 38.9597594, "lon": 34.9249653},

                           opacity=0.5,

                           labels={'unemp':'unemployment rate'}

                          )

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()