import os
print(os.listdir("../input"))
import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
import folium
from folium import plugins
from io import StringIO
init_notebook_mode(connected=True)
speed_cam_data = pd.read_csv("../input/speed-camera-locations.csv")
speed_cam_data.head()

maps = folium.Map(location=[41.878, -87.62], height = 700, tiles='Stamen Terrain', zoom_start=12)
for i in range(0,len(speed_cam_data)):
    folium.Marker([speed_cam_data.iloc[i]['LATITUDE'], speed_cam_data.iloc[i]['LONGITUDE']], popup=speed_cam_data.iloc[i]['ADDRESS']).add_to(maps)
maps
speed_cam_v_data = pd.read_csv("../input/speed-camera-violations.csv")
speed_cam_v_data.head()
import datetime
speed_cam_v_data['Violation date'] = pd.to_datetime(speed_cam_v_data['VIOLATION DATE'])
vuio_cnt = speed_cam_v_data.groupby(['ADDRESS', 'LATITUDE', 'LONGITUDE'])['VIOLATIONS'].size().sort_values(ascending=False).reset_index()
#vuio_cnt.head(5)
vio_maps = folium.Map(location=[41.878, -87.62], height = 700, tiles='OpenStreetMap', zoom_start=12)
for i in range(0,len(vuio_cnt)):
    folium.Marker([vuio_cnt.iloc[i]['LATITUDE'], vuio_cnt.iloc[i]['LONGITUDE']], popup=vuio_cnt.iloc[i]['ADDRESS'], icon=folium.Icon(color= 'red' if vuio_cnt.iloc[i]['VIOLATIONS'] > 1000 else 'green', icon='circle')).add_to(vio_maps)
#vio_maps
legend_html = '''
     <div style=”position: fixed; 
     bottom: 50px; left: 50px; width: 100px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     “>&nbsp;  <br>
     &nbsp; Green < 1000 Violations since july 2014 &nbsp; <i class=”fa fa-map-marker fa-2x”
                  style=”color:green”></i><br>
     &nbsp; Red > 1000 Violations since july 2014 &nbsp; <i class=”fa fa-map-marker fa-2x”
                  style=”color:red”></i>
      </div>
     '''
vio_maps.get_root().html.add_child(folium.Element(legend_html))
vio_maps
red_cam_data = pd.read_csv("../input/red-light-camera-violations.csv")
#red_cam_data.head()
spe_cnt = red_cam_data.groupby(['ADDRESS', 'LATITUDE', 'LONGITUDE'])['VIOLATIONS'].size().sort_values(ascending=False).reset_index()
#spe_cnt.head(5)
spe_maps = folium.Map(location=[41.878, -87.62], height = 700, tiles='stamenwatercolor', zoom_start=12)
for i in range(0,len(spe_cnt)):
    folium.Marker([spe_cnt.iloc[i]['LATITUDE'], spe_cnt.iloc[i]['LONGITUDE']], popup=spe_cnt.iloc[i]['ADDRESS'], icon=folium.Icon(color= 'red' if spe_cnt.iloc[i]['VIOLATIONS'] > 2000 else 'green', icon='circle')).add_to(spe_maps)
#spe_maps
legend_html = '''
     <div style=”position: fixed; 
     bottom: 50px; left: 50px; width: 100px; height: 90px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     “>&nbsp;  <br>
     &nbsp; Green < 2000 Violations Since July 2014 &nbsp; <i class=”fa fa-map-marker fa-2x”
                  style=”color:green”></i><br>
     &nbsp; Red > 2000 Violations Since July 2014 &nbsp; <i class=”fa fa-map-marker fa-2x”
                  style=”color:red”></i>
      </div>
     '''
spe_maps.get_root().html.add_child(folium.Element(legend_html))
spe_maps
red_cam_data_s = pd.read_csv("../input/red-light-camera-locations.csv")
#red_cam_data_s.head()
red_cam_maps = folium.Map(location=[41.878, -87.62], height = 700, tiles='CartoDB dark_matter', zoom_start=12)
for i in range(0,len(red_cam_data_s)):
    folium.CircleMarker([red_cam_data_s.iloc[i]['LATITUDE'], red_cam_data_s.iloc[i]['LONGITUDE']], radius=5, color='red', fill=True).add_to(red_cam_maps)
red_cam_maps
