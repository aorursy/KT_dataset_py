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
df=pd.read_csv('../input/procurement-notices/procurement-notices.csv')
df.head()
df['Publication Date']=pd.to_datetime(df['Publication Date'])
df['Deadline Date']=pd.to_datetime(df['Deadline Date'])


from datetime import date
df['Deadline Date'][df['Deadline Date']>date.today()].count()
count_dist=df[df['Deadline Date']>date.today()].groupby('Country Name').ID.count().reset_index(name='count')
count_dist
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import plotly.offline as py
from plotly import graph_objs as go
import folium
m = folium.Map(
    location = [12, 12],
    zoom_start = 2,
    tiles = 'CartoDB positron'
)

m.choropleth(
    geo_data = os.path.join('../input/worldcountries1', 'world-countries.json'),
    data = count_dist,
    columns = count_dist.columns,
    key_on = 'feature.properties.name',
    fill_color = 'YlOrRd',
    line_opacity = 0.5,
    fill_opacity = 0.8,
    smooth_factor = 1.0
    
)
m

df[df['Deadline Date']>date.today()].groupby('Deadline Date').ID.count().plot()

