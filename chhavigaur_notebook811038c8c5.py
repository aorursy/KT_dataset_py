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
df = pd.read_csv('/kaggle/input/gtd/globalterrorismdb_0718dist.csv', encoding='windows-1252')

df.head()
df.columns.to_list()
df['country_txt'].unique()
ind = df[df['country_txt'] == 'India']
ind.head()
df1 = df.copy()
df1.groupby('country_txt')
df1.head()
worst_hit = df[df['country_txt'].sum()]
worst_hit
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster
df3 = df[df['country_txt'] == 'India']
df4 = df3[['eventid','iyear','imonth', 'iday', 'latitude', 'longitude']]
df4
#df3.dropna(inplace = True)
null = df4.isna().sum()
null
df4.dropna(inplace = True)
df4

null[14]
m_1 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=10)
for ids, row in df4.iterrows():
    Marker([row["latitude"], row['longitude']]).add_to(m_1)
    
    
m_1






m_2 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = df4[['latitude', 'longitude']], radius = 10).add_to(m_2)
m_2
df4
df5 = df4.loc[(df4['iyear']>1970) & (df4['iyear']<=1980)]
df5
m_3 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = df5[['latitude', 'longitude']], radius = 10).add_to(m_3)
m_3
df6 = df4.loc[(df4['iyear']>1980) & (df4['iyear']<=1990)]
df6
m_4 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = df6[['latitude', 'longitude']], radius = 10).add_to(m_4)
m_4
df7 = df4.loc[(df4['iyear']>1990) & (df4['iyear']<=2000)]
df7
m_5 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = df7[['latitude', 'longitude']], radius = 10).add_to(m_5)
m_5
df8 = df4.loc[(df4['iyear']>2000) & (df4['iyear']<=2010)]
df8
m_6 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = df8[['latitude', 'longitude']], radius = 10).add_to(m_6)
m_6
df9 = df4.loc[(df4['iyear']>2010) & (df4['iyear']<=2020)]
df9
m_7 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = df9[['latitude', 'longitude']], radius = 10).add_to(m_7)
m_7
world_terr = df[['eventid','iyear','imonth', 'iday', 'latitude', 'longitude']]
world_terr
world_terr.dropna(inplace = True)
m_8 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = world_terr[['latitude', 'longitude']], radius = 5).add_to(m_8)
m_8
df10 = world_terr.loc[(world_terr['iyear']>2010) & (world_terr['iyear']<=2020)]
df10
m_9 = folium.Map(location=[28.585836, 77.153336], tiles='cartodbpositron', zoom_start=12)
HeatMap(data = df10[['latitude', 'longitude']], radius = 3).add_to(m_9)
m_9
