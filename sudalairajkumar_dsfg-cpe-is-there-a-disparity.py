import os
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# folium for maps
import folium
from folium import plugins

# geopandas for operations on shape files
import geopandas as gpd
from shapely.geometry import Polygon
from pprint import pprint 

# plotly for other visuals
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
import plotly.plotly as py
from plotly import tools
import plotly.figure_factory as ff
init_notebook_mode(connected=True)
pd.options.display.max_columns = 999
os.listdir("../input/data-science-for-good/cpe-data/")
os.listdir("../input/data-science-for-good/cpe-data/Dept_24-00013")
os.listdir("../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_ACS_data/")
fname = "../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_ACS_data/24-00013_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv"
acs_race_df = pd.read_csv(fname)
acs_race_clean_df = acs_race_df.loc[1:,:].reset_index(drop=True)

acs_race_clean_df["CT"] = acs_race_clean_df["GEO.display-label"].apply(lambda x: x.split("Tract ")[1].split(",")[0].strip())

acs_race_df.head()
force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_UOF_2008-2017_prepped.csv")
force_clean_df = force_df.loc[1:,:].reset_index(drop=True)
force_df.head()
fname = "../input/data-science-for-good/cpe-data/Dept_24-00013/24-00013_Shapefiles/Minneapolis_Police_Precincts.shp"
police_df = gpd.read_file(fname)
police_df.head()
mapa = folium.Map([44.99, -93.27], height=500, zoom_start=11, tiles='Stamen Terrain')
folium.GeoJson(police_df).add_to(mapa)
mapa
mapa = folium.Map([44.99, -93.27], height=500, zoom_start=11, tiles='Stamen Terrain')

folium.GeoJson(police_df).add_to(mapa)

locations_df = force_clean_df[["LOCATION_LATITUDE", "LOCATION_LONGITUDE"]].copy()
notna = locations_df[['LOCATION_LATITUDE','LOCATION_LONGITUDE']].dropna().index
locations_df = locations_df.iloc[notna].reset_index(drop=True)
locations_df["LOCATION_LATITUDE"] = locations_df["LOCATION_LATITUDE"].astype('float')
locations_df["LOCATION_LONGITUDE"] = locations_df["LOCATION_LONGITUDE"].astype('float')
locationlist = locations_df.values.tolist()[-2000:]
for point in range(0, len(locationlist)):
    #folium.Marker(locationlist[point], popup=df_counters['Name'][point], icon=folium.Icon(color='darkblue', icon_color='white', icon='male', angle=0, prefix='fa')).add_to(marker_cluster)
    folium.CircleMarker(locationlist[point], radius=0.1, color='red').add_to(mapa)

mapa 
f, ax = plt.subplots(1, figsize=(10, 8))
police_df.plot(column="PRECINCT", ax=ax, cmap='Accent',legend=True);
plt.title("Districts : Minneapolis Police Precincts")
plt.show()
fname = "../input/dsfg-cpe-acs-shape-files/cb_2017_27_tract_500k/cb_2017_27_tract_500k.shp"
acs_df = gpd.read_file(fname)
acs_df = acs_df[acs_df["COUNTYFP"]=="053"].reset_index()
acs_df.head()
mapa = folium.Map([45.04, -93.47], height=600, zoom_start=10, tiles='Stamen Terrain')
folium.GeoJson(acs_df).add_to(mapa)
mapa
mapa = folium.Map([44.98, -93.27], height=500, zoom_start=13, tiles='Stamen Terrain')
folium.GeoJson(police_df.loc[0:0,:], style_function= lambda x:{'color':'red'}).add_to(mapa)
mapa
police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][0]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]
mapa = folium.Map([44.98, -93.27], height=500, zoom_start=13, tiles='Stamen Terrain')
folium.GeoJson(police_df.loc[0:0,:], style_function= lambda x:{'color':'red'}).add_to(mapa)
folium.GeoJson(acs_df[acs_df["NAME"].isin(acs_police_df["NAME"].values)], style_function= lambda x:{'color':'green'}).add_to(mapa)
mapa
# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]=="1"].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", "Native American":"Red"}
color_names = []
for i in labels:
    color_names.append(color_map[i])

trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District 1',
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')


police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 1

#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

mapa = folium.Map([44.98, -93.27], height=500, zoom_start=12, tiles='Stamen Terrain')
folium.GeoJson(police_df.loc[POLICE_DIST_ROW:POLICE_DIST_ROW,:], style_function= lambda x:{'color':'red'}).add_to(mapa)
folium.GeoJson(acs_df[acs_df["NAME"].isin(acs_police_df["NAME"].values)], style_function= lambda x:{'color':'green'}).add_to(mapa)
mapa
# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "2"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", "Native American":"Red"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')

police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 2
#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "3"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')
police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 3
#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "4"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')
police_gdf = gpd.GeoDataFrame(police_df["geometry"])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

POLICE_DIST_ROW = 4
#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf['geometry'][POLICE_DIST_ROW]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
POLICE_DISTRICT = "5"
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==POLICE_DISTRICT].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='Minneapolis Police District '+POLICE_DISTRICT,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')
os.listdir("../input/data-science-for-good/cpe-data/Dept_24-00098/")
### American community survey data
fname = "../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_ACS_data/24-00098_ACS_race-sex-age/ACS_15_5YR_DP05_with_ann.csv"
acs_race_df = pd.read_csv(fname)
acs_race_clean_df = acs_race_df.loc[1:,:].reset_index(drop=True)
acs_race_clean_df["CT"] = acs_race_clean_df["GEO.display-label"].apply(lambda x: x.split("Tract ")[1].split(",")[0].strip())

force_df = pd.read_csv("../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Vehicle-Stops-data.csv")
force_clean_df = force_df.loc[1:,:].reset_index(drop=True)

fname = "../input/data-science-for-good/cpe-data/Dept_24-00098/24-00098_Shapefiles/StPaul_geo_export_6646246d-0f26-48c5-a924-f5a99bb51c47.shp"
police_df = gpd.read_file(fname)

#-93.18, 44.95
mapa = folium.Map([44.99, -93.08], height=500, zoom_start=11, tiles='Stamen Toner')
folium.GeoJson(police_df).add_to(mapa)
mapa
fname = "../input/dsfg-cpe-acs-shape-files/cb_2017_27_tract_500k/cb_2017_27_tract_500k.shp"
acs_df = gpd.read_file(fname)
acs_df = acs_df[acs_df["COUNTYFP"]=="123"].reset_index()
acs_df.head()

mapa = folium.Map([44.99, -93.08], height=600, zoom_start=11, tiles='Stamen Toner')
folium.GeoJson(acs_df).add_to(mapa)
mapa
### Config ###
police_area_column = "gridnum"
police_area_value = "1"
police_shp_column = "geometry"

police_district = "1"
police_gdf = gpd.GeoDataFrame(police_df[police_shp_column])
acs_gdf = gpd.GeoDataFrame(acs_df["geometry"])

#print("Following Census Tracts are present in the Minneapolis Police District 1 : ")
acs_police_df = []
for i in range(acs_gdf.shape[0]):
    a = (police_gdf[police_shp_column][police_df[police_area_column]==police_area_value].iloc[0]).intersection(acs_gdf['geometry'][i])
    if a.area != 0:
        #print("CT :", acs_df['NAME'][i], " and the percentage of area is :", (a.area / acs_gdf['geometry'][i].area)*100)
        acs_police_df.append([acs_df['NAME'][i], (a.area / acs_gdf['geometry'][i].area)])
acs_police_df = pd.DataFrame(acs_police_df)
acs_police_df.columns = ["NAME", "PercentageArea"]

# pie chart based on population
acs_police_race_df = acs_race_clean_df[acs_race_clean_df["CT"].isin(acs_police_df["NAME"].values)].reset_index(drop=True)
acs_police_race_df = pd.merge(acs_police_race_df, acs_police_df, left_on=["CT"], right_on=["NAME"])

cols_to_use = ["HC01_VC49", "HC01_VC50", "HC01_VC51", "HC01_VC56", "HC01_VC69"]
col_names = ["White", "Black", "Native American", "Asian", "Other / Mixed Race"]
color_names = ["LightGrey", "Black", "Red", "Orange", "Green"]

race_count = []
for i, col in enumerate(cols_to_use):
    race_count.append([col_names[i], np.round((acs_police_race_df[col].astype(float) * acs_police_race_df["PercentageArea"]).sum())])
race_count_df = pd.DataFrame(race_count)
race_count_df.columns = ["race", "count"]

labels = (np.array(race_count_df["race"].values))
temp_series = race_count_df["count"]
sizes = (np.array((temp_series / temp_series.sum())*100))

trace0 = go.Pie(labels=labels, 
                values=sizes,
                domain=dict(x=[0,0.48]),
                marker=dict(
                   colors=color_names
               ),
              )

## Pie chart based on use of force
temp_series = force_clean_df["SUBJECT_RACE"][force_clean_df["LOCATION_DISTRICT"]==police_district].value_counts().head(5)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))
color_map = {"Black":"Black", "White":"LightGrey", "Other / Mixed Race":"Green", "Asian":"Orange", 
             "Native American":"Red", "not recorded":"blue", "Latino":"green", "No Data":"yellow"}
color_names = []
for i in labels:
    color_names.append(color_map[i])
    
trace1 = go.Pie(labels=labels, 
               values=sizes,
                domain=dict(x=[0.52,1]),
               marker=dict(
                   colors=color_names
               ),
              )
ann1 = dict(font=dict(size=12),
            showarrow=False,
            text='Census Population Distribution by Race ',
            # Specify text position (place text in a hole of pie)
            x=0.1,
            y=1.1,
            )
ann2 = dict(font=dict(size=12),
            showarrow=False,
            text='Use of Force Distribution by Race',
            # Specify text position (place text in a hole of pie)
            x=0.9,
            y=1.1,
            )
layout = go.Layout(title ='St Paul Police District '+police_district,
                   annotations=[ann1,ann2],
                   # Hide legend if you want
                   #showlegend=False
                   )

data = [trace0, trace1]
fig = go.Figure(data=data,layout=layout)
# Plot the plot and save the file in your Python script directory
iplot(fig, filename='subplot_pie_chart.html')

