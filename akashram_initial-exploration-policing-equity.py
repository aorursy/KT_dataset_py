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
departs = os.listdir("../input/cpe-data/")
departs
files = os.listdir("../input/cpe-data/Dept_35-00103/35-00103_ACS_data/")
files
path_dept_35 = "../input/cpe-data/Dept_35-00103/35-00103_ACS_data/35-00103_ACS_poverty/"
pov_df = pd.read_csv(path_dept_35 + "ACS_16_5YR_S1701_with_ann.csv")
pov_df.head()
# pov_df.iloc[0].unique()
# pov_df['GEO.display-label'].unique()
total_pov_estimate_population = pov_df["HC01_EST_VC01"][1:]

trace = go.Histogram(x=total_pov_estimate_population, marker=dict(color='blue', opacity=0.8))
layout = dict(title="Total Estimate for Poverty - Distribution across Mecklenburg County, North Carolina", margin=dict(l=200), width=1000, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)

total_bpov_estimate_population = pov_df["HC02_EST_VC01"][1:]

trace = go.Histogram(x=total_bpov_estimate_population, marker=dict(color='Red', opacity=0.8))
layout = dict(title="Total Estimate for Below Poverty - Distribution across Mecklenburg County, North Carolina", margin=dict(l=200), width=1000, height=400)
data = [trace]
fig = go.Figure(data=data, layout=layout)
iplot(fig)
pov_meta_df = pd.read_csv(path_dept_35 + "ACS_16_5YR_S1701_metadata.csv")
pov_meta_df.head()
trace1 = go.Box(x = pov_df["HC01_EST_VC03"][1:])#, name="18+", marker=dict(opacity=0.4)) 
trace2 = go.Box(x = pov_df["HC01_EST_VC04"][1:])#, name="<5", marker=dict(opacity=0.3)) 
trace3 = go.Box(x = pov_df["HC01_EST_VC05"][1:])#, name="5-17", marker=dict(opacity=0.4))
trace4 = go.Box(x = pov_df["HC01_EST_VC08"][1:])#, name="18-34", marker=dict(opacity=0.4))
trace5 = go.Box(x = pov_df["HC01_EST_VC07"][1:])#, name="18-64", marker=dict(opacity=0.4))

titles = ["Age : 18+","Age : <5","Age: 5-17","Age : 18-34", "Age: 18-64"]
fig = tools.make_subplots(rows=3, cols=2, print_grid=False, subplot_titles=titles)
fig.append_trace(trace1, 1, 1);
fig.append_trace(trace2, 1, 2);
fig.append_trace(trace3, 2, 1);
fig.append_trace(trace4, 2, 2);
fig.append_trace(trace5, 3, 1);
fig['layout'].update(height=600, title="Distribution of Age-Poverty across Mecklenburg County, North Carolina");
iplot(fig, filename='age-subplots');
map1 = "../input/cpe-data/Dept_35-00103/35-00103_Shapefiles/CMPD_Police_Division_Offices.shp"
nc_maps = gpd.read_file(map1) 
nc_maps_df = pd.DataFrame(nc_maps)
nc_maps_df['geometry'] = nc_maps_df['geometry'].astype(str)
nc_maps_df['geometry'] = nc_maps_df['geometry'].str.replace("POINT", "")
nc_maps_df['geometry'] = nc_maps_df['geometry'].str.replace("(", "")
nc_maps_df['geometry'] = nc_maps_df['geometry'].str.replace(")", "")
nc_maps_df["Latitude"], nc_maps_df["Longitude"] = zip(*nc_maps_df["geometry"].str.split().tolist())

nc_maps_df[['Longitude', 'Latitude']] = nc_maps_df[['Longitude','Latitude']].apply(pd.to_numeric)
maps = folium.Map(location=[35.2633, -80.85], height = 700, tiles='Stamen Terrain', zoom_start=12)
for i in range(0,len(nc_maps_df)):
    folium.Marker([nc_maps_df.iloc[i]['Longitude'], nc_maps_df.iloc[i]['Latitude']], popup=nc_maps_df.iloc[i]['STNAME']).add_to(maps)
maps
nc_uof =  pd.read_csv("../input/cpe-data/Dept_35-00103/35-00103_UOF-OIS-P_prepped.csv")
nc_uof = nc_uof.drop(nc_uof.index[0])
nc_uof = nc_uof.drop(nc_uof.index[68])
nc_uof[['LOCATION_LATITUDE', 'LOCATION_LONGITUDE']] = nc_uof[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].apply(pd.to_numeric)
maps1 = folium.Map(location=[35.2633, -80.85], height = 700, tiles='Stamen Toner', zoom_start=12)
for i in range(0,len(nc_uof)):
    folium.Marker([nc_uof.iloc[i]['LOCATION_LATITUDE'], nc_uof.iloc[i]['LOCATION_LONGITUDE']], popup=nc_uof.iloc[i]['LOCATION_FULL_STREET_ADDRESS_OR_INTERSECTION'], icon=folium.Icon(color= 'black' if nc_uof.iloc[i]['SUBJECT_RACE'] == 'Black' else 'red', icon='circle')).add_to(maps1)
maps1
map3 = "../input/cpe-data/Dept_23-00089/23-00089_Shapefiles/Indianapolis_Police_Zones.shp"
idp_maps = gpd.read_file(map3) 
idp_maps_df = pd.DataFrame(idp_maps)
idp_maps_df['geometry'] = idp_maps_df['geometry'].astype(str)
idp_maps_df['geometry'] = idp_maps_df['geometry'].str.replace("POLYGON ", "")
idp_maps_df['geometry'] = idp_maps_df['geometry'].str.replace("(", "")
idp_maps_df['geometry'] = idp_maps_df['geometry'].str.replace(")", "")
idp_maps_df['geometry'] = idp_maps_df['geometry'].str.replace("MULTI", "")
idp_maps_df["Latitude"] = idp_maps_df.geometry.str.split(' ').str[0].tolist()
idp_maps_df["Longitude"] = idp_maps_df.geometry.str.split(' ').str[1].tolist()
idp_maps_df['Longitude'] = idp_maps_df['Longitude'].str.replace(",", "")
idp_maps_df[['Longitude', 'Latitude']] = idp_maps_df[['Longitude','Latitude']].apply(pd.to_numeric)
maps_ind = folium.Map(location=[39.81, -86.26060805912148], height = 700, tiles='Stamen Terrain', zoom_start=12)
for i in range(0,len(idp_maps_df)):
    folium.Marker([idp_maps_df.iloc[i]['Longitude'], idp_maps_df.iloc[i]['Latitude']], popup=idp_maps_df.iloc[i]['POLICEZONE']).add_to(maps_ind)
maps_ind
map4 = "../input/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp"
apd_maps = gpd.read_file(map4) 
apd_maps_df = pd.DataFrame(apd_maps)
map5 = "../input/cpe-data/Dept_37-00049/37-00049_Shapefiles/EPIC.shp"
epic_maps = gpd.read_file(map5) 
epic_maps_df = pd.DataFrame(epic_maps)
map5 = "../input/cpe-data/Dept_49-00009/49-00009_Shapefiles/SPD_BEATS_WGS84.shp"
spd_maps = gpd.read_file(map5) 
spd_maps_df = pd.DataFrame(spd_maps)
map2 = "../input/cpe-data/Dept_11-00091/11-00091_Shapefiles/boston_police_districts_f55.shp"
bos_maps = gpd.read_file(map2) 
bos_maps_df = pd.DataFrame(bos_maps)