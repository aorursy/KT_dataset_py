# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px #graphic library 

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df=pd.read_csv('/kaggle/input/geospatial-sao-paulo-crime-database/dataset-limpo.csv')
df.head()
df.columns

fig = px.density_mapbox(df, lat='latitude', lon='longitude', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain")
fig.update_layout(
        title = 'Mapbox Density Heatmap - Crimes',
)
fig.show()
df['time']
df['date']=pd.to_datetime(df['time'], format='%Y-%m-%d').dt.year
df=df.sort_values('date')
fig = px.density_mapbox(df, lat='latitude', lon='longitude', radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain", animation_frame='date')
fig.update_layout(
        title = 'Mapbox Density Heatmap - Crimes',
)
fig.show()

import plotly.express as px
fig = px.density_mapbox(df, lat='latitude', lon='longitude', z='valor_prejuizo',
                        radius=5, zoom=7,
                        color_continuous_scale=px.colors.sequential.YlOrRd,
                        mapbox_style="stamen-terrain", animation_frame='date')
fig.update_layout(
        title = 'Mapbox Density Heatmap - Prejuizo',
)
fig.show()