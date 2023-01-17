import geopandas as gpd
import matplotlib.pyplot as plt
import folium

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
dist = gpd.read_file('../input/india-district-wise-shape-files/output.shp')
dist.head()
dist.isna().sum()
# Handling NA values
dist.distarea.fillna(dist.distarea.mean(), inplace=True)
dist.totalpopul.fillna(dist.totalpopul.mean(), inplace=True)
dist.totalhh.fillna(dist.totalhh.mean(), inplace=True)
dist.totpopmale.fillna(dist.totpopmale.mean(), inplace=True)
dist.totpopfema.fillna(dist.totpopfema.mean(), inplace=True)
dist.isna().sum()
dist.plot(figsize=(10,10))
plt.show()
dist.plot(figsize=(10,10), cmap='inferno', column='totalpopul', legend=True)
statename='Madhya Pradesh'
state_dist = dist[dist.statename == statename]
ax=state_dist.plot(cmap='coolwarm', column=['totalpopul'], scheme='Percentiles',figsize=(10, 10), legend=True)
plt.show()
states = dist.dissolve(by='statename',aggfunc='sum').reset_index()

# New Column SexRation which gives number of females per 1000 males 
states["SexRatio"]=states.totpopfema*1000/states.totpopmale
states.head()
states.plot(cmap='rainbow', figsize=(10,10), column='SexRatio', legend=True)
plt.show()
m = folium.Map(location=[23, 78.9629], tiles='cartodbpositron',
               min_zoom=4, max_zoom=6, zoom_start=4)

folium.Choropleth(geo_data=states, data=states, name='choropleth',
                  columns=['statename', 'SexRatio'], 
                  key_on='feature.properties.statename',                  
                  fill_color='PuRd',
                  line_weight=0.1,
                  line_opacity=0.5,
                  legend_name='Sex Ratio').add_to(m)

folium.LayerControl().add_to(m)

m
m = folium.Map(location=[23, 78.9629], tiles='cartodbpositron',
               min_zoom=4, max_zoom=6, zoom_start=4)

folium.Choropleth(geo_data=dist, data=dist, name='choropleth',
                  columns=['distname', 'totalpopul'], 
                  key_on='feature.properties.distname',                  
                  fill_color='YlGn',
                  line_weight=0.1,
                  line_opacity=0.5,
                  legend_name='Total Population').add_to(m)

folium.LayerControl().add_to(m)

m
