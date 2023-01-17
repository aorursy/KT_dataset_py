import pandas as pd

import folium



url = 'https://raw.githubusercontent.com/python-visualization/folium/master/examples/data'

state_geo = f'{url}/us-states.json'

state_unemployment = f'{url}/US_Unemployment_Oct2012.csv'

state_data = pd.read_csv(state_unemployment)



m = folium.Map(location=[48, -102], zoom_start=3)



folium.Choropleth(

    geo_data=state_geo,

    name='choropleth',

    data=state_data,

    columns=['State', 'Unemployment'],

    key_on='feature.id',

    fill_color='YlGn',

    fill_opacity=0.7,

    line_opacity=0.2,

    legend_name='Unemployment Rate (%)'

).add_to(m)



folium.LayerControl().add_to(m)



m
import os

import numpy as np 

import pandas as  pd



import folium

print('numpy ver: ', np.__version__)

print('pandas ver: ', pd.__version__)

print('folium ver: ', folium.__version__)
korea = folium.Map(location=[37, 126],

                   zoom_start=5)

korea
korea_black = folium.Map(location=[37, 126],

                   zoom_start=7,

                   tiles='Stamen Toner'

                  )

korea_black
korea.save('korea_map.html')
korea = folium.Map(location=[37.56, 126.97],

                   zoom_start=10,

                   tiles='Stamen Terrain'

 )



folium.Marker([37.56, 126.97], 

              popup='<i>The capital of Korea</i>', 

              tooltip='Seoul'

             ).add_to(korea)



folium.Marker([37.45, 126.70], 

              popup='<i>International Airport</i>', 

              tooltip='Incheon',

              icon=folium.Icon(icon='plane', color='red')

             ).add_to(korea)





korea
korea = folium.Map(location=[37.56, 126.97],

                   zoom_start=14,

                   tiles='Stamen Terrain'

 )



# Circle

folium.Circle(location=[37.56, 126.97],

              radius=200, # in meter

              fill=True,

              tooltip='this is Circle'

             ).add_to(korea)



# Circle Marker

folium.CircleMarker(location=[37.565, 126.98],

                    radius=80, # in pixel

                    fill=True,

                    color='red',

                    tooltip='this is Circle Marker'

                   ).add_to(korea)



folium.Rectangle(bounds=[(37.554, 126.95), (37.556, 126.97)],

                    fill=True,

                    color='orange',

                    tooltip='this is Rectangle'

                   ).add_to(korea)



korea
from folium import plugins
m = folium.Map([30, 0], zoom_start=3) 

plugins.BoatMarker( location=(34, -43), heading=45, wind_heading=150, wind_speed=45, color='#8f8' ).add_to(m) 

plugins.BoatMarker( location=(46, -30), heading=-20, wind_heading=46, wind_speed=25, color='#88f' ).add_to(m) 

m
world = folium.Map() # full map

draw = plugins.Draw(export=True)

draw.add_to(world)

world
korea = folium.Map(location=[37.56, 126.97],

                   zoom_start=10,

                   tiles='Stamen Terrain')



plugins.Fullscreen(position='topright', # ‘topleft’, default=‘topright’, ‘bottomleft’, ‘bottomright’ 

                   title='FULL SCREEN ON', 

                   title_cancel='FULL SCREEN OFF',

                   force_separate_button=True

                  ).add_to(korea)



korea
data = (np.random.normal(size=(100, 3)) * np.array([[1, 1, 1]]) + np.array([[48, 5, 1]]) ).tolist()



m = folium.Map([48., 5.],  zoom_start=6) 

plugins.HeatMap(data).add_to(m) 

m



N = 100 # number of marker



EU = folium.Map(location=[45, 3], 

                zoom_start=4)



points = np.array([

        np.random.uniform(low=35, high=60, size=N),  

        np.random.uniform(low=-12, high=30, size=N)]).T



plugins.MarkerCluster(points).add_to(EU)

EU
korea = folium.Map(location=[37.56, 126.97],

                   zoom_start=10,

                   tiles='Stamen Terrain')



minimap = plugins.MiniMap()

korea.add_child(minimap)



korea
m = folium.Map()



formatter = "function(num) {return L.Util.formatNum(num, 3) + ' º ';};"



plugins.MousePosition(

    position='topright',

    separator=' | ',

    empty_string='NaN',

    lng_first=True,

    num_digits=20,

    prefix='Coordinates:',

    lat_formatter=formatter,

    lng_formatter=formatter,

).add_to(m)



m
korea = folium.Map(location=[37.56, 126.97], 

                   zoom_start=1) 



plugins.Terminator().add_to(korea) 



korea