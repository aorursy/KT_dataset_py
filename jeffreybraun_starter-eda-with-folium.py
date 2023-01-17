import numpy as np
import pandas as pd
import seaborn as sns
import plotly as py
import folium 
from folium import plugins

df = pd.read_csv('/kaggle/input/chipotle-locations/chipotle_stores.csv')
df.head()
df.info()
usa_map = folium.Map([39.358, -98.118], zoom_start=4)
map_title = 'Chipotle Locations Heatmap'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(map_title) 
usa_map.get_root().html.add_child(folium.Element(title_html))
usa_map.add_child(plugins.HeatMap(df[['latitude', 'longitude']]))
usa_map

my_USA_map = '/kaggle/input/chipotle-locations/us-states.json'
df_state = pd.DataFrame(df.state.value_counts()).reset_index()
df_state.rename(columns = {'index':'state', 'state':'count'}, inplace=True)
state_map = folium.Map([39.358, -98.118], zoom_start=4)
map_title = 'Chipotle Locations by State'
title_html = '''
             <h3 align="center" style="font-size:16px"><b>{}</b></h3>
             '''.format(map_title) 
state_map.get_root().html.add_child(folium.Element(title_html))
state_map.choropleth(geo_data=my_USA_map, data=df_state,
             columns=['state', 'count'],
             key_on='feature.properties.name',
             fill_color='YlGn', fill_opacity=0.7, line_opacity=0.2,
             legend_name='Number of Chipotles')
state_map