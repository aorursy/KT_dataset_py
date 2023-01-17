# Import necessary libraries

import geopandas as gdp

import numpy as np

import pandas as pd



import osmnx as ox

%matplotlib inline

print("OSMNX version:", ox.__version__)



import folium

from folium import Choropleth, Marker

from folium.plugins import TimeSliderChoropleth, Fullscreen, MiniMap, LocateControl



from PIL import Image



print("Folium version:", folium.__version__)
# places = ['1st district, Budapest, Hungary', '2nd district, Budapest, Hungary', '3rd district, Budapest, Hungary',

#           '4th district, Budapest, Hungary', '5th district, Budapest, Hungary', '6th district, Budapest, Hungary',

#           '7th district, Budapest, Hungary', '8th district, Budapest, Hungary', '9th district, Budapest, Hungary',

#           '10th district, Budapest, Hungary', '11th district, Budapest, Hungary', '12th district, Budapest, Hungary',

#           '13th district, Budapest, Hungary', '14th district, Budapest, Hungary', '15th district, Budapest, Hungary',

#           '16th district, Budapest, Hungary', '17th district, Budapest, Hungary', '18th district, Budapest, Hungary', 

#           '19th district, Budapest, Hungary', '20th district, Budapest, Hungary', '21st district, Budapest, Hungary',

#           '22nd district, Budapest, Hungary', '23rd district, Budapest, Hungary']



# gdf_districts = ox.gdf_from_places(places)
gdf_districts = gdp.read_file("../input/geo-bud-districts/Gdf_districts/Gdf_districts/gdf_districts.shp")



for i in range(len(gdf_districts)):

    gdf_districts.loc[i,'district_code'] = i + 1

    

gdf_districts.head()
gdf_districts["centroid"] = gdf_districts.geometry.centroid

gdf_districts.head()
station_in_XVIII = [47.431369, 19.182132]

station_in_I = [47.508139, 19.027195]

station_in_VIII = [47.493856, 19.084479]

station_in_II = [47.562394, 18.961156]

station_in_IV = [47.585405, 19.114884]

station_in_XV = [47.543465, 19.146288]

station_in_XI = [47.475782, 19.041205]

station_in_XIII = [47.521744, 19.068248]

# station_in_V = [47.497555, 19.052645]

station_in_X = [47.467456, 19.155798]

station_in_XXII = [47.406248, 19.009319]

station_in_XXI = [47.404750, 19.091043]
list_of_stations = [station_in_XVIII, station_in_I, station_in_VIII, station_in_II, station_in_IV, 

                    station_in_XV, station_in_XI, station_in_XIII, station_in_X, station_in_XXII, station_in_XXI]



# Change places latitude and longitude as required further

for station in list_of_stations:

    station[0], station[1] = station[1], station[0]
stations_df = pd.DataFrame(list_of_stations, columns = ["latitude","longitude"])



names = ['station_in_XVIII', 'station_in_I', 'station_in_VIII', 'station_in_II', 

         'station_in_IV', 'station_in_XV', 'station_in_XI', 'station_in_XIII', 'station_in_X',

         'station_in_XXII', 'station_in_XXI']

stations_df['station_district'] = names



gdf_stations = gdp.GeoDataFrame(stations_df, geometry=gdp.points_from_xy(stations_df.latitude, stations_df.longitude))

gdf_stations.head()
# Assign the same CRS to both dataframes (EPSG 4326 that uses latitude and longitude)

gdf_stations.crs = {'init' :'epsg:4326'}

gdf_districts.crs = {'init' :'epsg:4326'}



# Initialize a map

m_stations = folium.Map(location=[47.4917,19.1102], min_zoom=2, max_bounds=True, tiles='cartodbpositron', zoom_start=11)



# Add a GeoJsonTooltip to the map to show district number when hovering the mouse over it

mykwargs = {"fields": ['district_code']} # works

# mykwargs = {"permanent": True, "fields": ['district_code']} # does not work

mytooltip = folium.GeoJsonTooltip(**mykwargs)



# Draw a simple map with district borders

folium.GeoJson(

    gdf_districts[['geometry','district_code']], 

    tooltip=mytooltip,

    style_function=lambda x: {'color':'black','fillColor':'green','weight':1},

    highlight_function=lambda x: {'weight':3,'fillColor':'yellow'}

    ).add_to(m_stations)



# Add stations to the map

for idx, row in gdf_stations.iterrows():

    Marker([row['longitude'], row['latitude']],popup=row['station_district']).add_to(m_stations)



# Add tile layers to the map

tiles = ["Stamen Toner", 'stamenwatercolor', 'cartodbpositron', 'openstreetmap', 'stamenterrain', "Mapbox Bright"]

for tile in tiles:

    folium.TileLayer(tile).add_to(m_stations)



# Create a layer control

folium.LayerControl(collapsed=True).add_to(m_stations)



# Save and display the map

m_stations.save('stations_and_borders.html')

m_stations
gdf_centroid = gdp.GeoDataFrame(gdf_districts["district_code"], geometry=gdf_districts["centroid"])

gdf_centroid.head()
gdf_stations.crs = {'init' :'epsg:2263'}

gdf_centroid.crs = {'init' :'epsg:2263'}
closest_stations = []



for district in range(len(gdf_centroid)):

    distance_list = gdf_stations.geometry.distance(gdf_centroid.iloc[district].geometry)

    closest_stations.append(gdf_stations.iloc[distance_list.idxmin()]["station_district"])

    

gdf_districts["closest_station"] = closest_stations

gdf_districts.drop(["bbox_north","bbox_south","bbox_east","bbox_west"], axis=1, inplace=True)

gdf_districts.head()
AQI_raw = pd.read_csv("../input/geo-bud-districts/AQI_raw_no_st_V.csv",index_col=[0], header=[0,1],skip_blank_lines=False)



AQI_raw.columns.set_levels(['station_in_I','station_in_II',"station_in_IV",'station_in_VIII','station_in_X','station_in_XI',

                           'station_in_XIII','station_in_XV','station_in_XVIII','station_in_XXI','station_in_XXII'],level=0,inplace=True)

AQI_raw = AQI_raw.apply(pd.to_numeric,errors='coerce')

AQI_raw.head()
AQI_raw = AQI_raw.bfill().fillna(AQI_raw.median())

AQI_raw.isnull().values.any()
unstacked_AQI = AQI_raw.unstack()

unstacked_AQI.index.names = ["station_in_district", "parameter", "date"]

unstacked_AQI.head()
median_AQI_per_param = unstacked_AQI.groupby(["station_in_district","parameter"]).median()

median_AQI_per_param.head()
historical_AQI_per_station = median_AQI_per_param.groupby(["station_in_district"]).max()

historical_AQI_per_station.head()
gdf_districts['AQI']=gdf_districts.closest_station.map(historical_AQI_per_station)



gdf_districts['district_id'] = ['district_I','district_II','district_III','district_IV','district_V','district_VI','district_VII',

                            'district_VIII','district_IX','district_X','district_XI','district_XII','district_XIII','district_XIV',

                            'district_XV','district_XVI','district_XVII','district_XVIII','district_XIX','district_XX','district_XXI',

                            'district_XXII','district_XXIII']

gdf_districts.set_index('district_id', inplace=True)



gdf_districts.head()
districts = gdp.GeoDataFrame(gdf_districts[['geometry','district_code']])

districts.head()
# Initialize a map

m_choropleth_AQI = folium.Map(location=[47.4917,19.0902], min_zoom=2, max_bounds=True, tiles='cartodbpositron', zoom_start=11)



# Draw a choropleth map on it

choropleth = Choropleth(geo_data=districts.to_json(),

                        data=gdf_districts.AQI,

                        key_on="feature.id", 

                        fill_color='YlGnBu', 

                        legend_name='Air Quality Index - the max median among 5 parameters over 2016-2019',

                        highlight = True

             ).add_to(m_choropleth_AQI)



# Add a GeoJsonTooltip to the map to show district number when hovering the mouse over it

choropleth.geojson.add_child(folium.features.GeoJsonTooltip(['district_code'],labels=False)

)



# Add a fullscreen view tool to the top left (not working on Kaggle, works if saved and opened as html)

m_choropleth_AQI.add_child(Fullscreen(position='topleft', title='Full Screen', title_cancel='Exit Full Screen'))



# Add a minimap in the bottom right corner. Moving or zooming the small map affect the big one also. 

# Here just to show this functionality, no real purpose.

minimap = MiniMap(tile_layer='cartodbpositron',zoom_level_offset=-6)

m_choropleth_AQI.add_child(minimap)





# Add a LocateControl tool to the top left (not working on Kaggle, works if saved and opened as html).

# By clicking on it, it will show on the map the geolocation of the user.

LocateControl(auto_start=True).add_to(m_choropleth_AQI)



# Save and display the map

m_choropleth_AQI.save('median_AQI_per_district.html')

m_choropleth_AQI
i = Image.open("../input/geo-bud-districts/Description of AQI levels.png")

i
daily_AQI = unstacked_AQI.groupby(["station_in_district","date"]).max()



# Log transform AQI index values to show the differences in time more clearly

daily_AQI = np.log(daily_AQI)
daily_AQI.index = daily_AQI.index.set_levels((pd.to_datetime(daily_AQI.index.levels[1])

                 .astype(int) // 10**9)

                 .astype('U10'), level=1)

daily_AQI.head()
max_AQI = daily_AQI.max()

max_AQI
min_AQI = daily_AQI.min()

min_AQI
from branca.colormap import linear



# abot color brewer

cmap = linear.YlOrRd_09.scale(min_AQI, max_AQI)

daily_AQI = daily_AQI.apply(cmap)
daily_AQI_df = pd.DataFrame(daily_AQI, columns=['color'])

daily_AQI_df.reset_index(level='date',inplace=True)

daily_AQI_df.head()
closest_stations = pd.DataFrame(gdf_districts[['closest_station','district_code']]).reset_index(drop=False)

closest_stations.head()
style_df = closest_stations.merge(daily_AQI_df[['date','color']], left_on='closest_station', right_on='station_in_district')

style_df.drop('closest_station', axis=1, inplace=True)

style_df.head()
district_list = style_df['district_id'].unique().tolist()

district_idx = range(len(district_list))



style_dict = {}

for i in district_idx:

    district = district_list[i]

    result = style_df[style_df['district_id'] == district]

    inner_dict = {}

    for _, r in result.iterrows():

        inner_dict[r['date']] = {'color': r['color'], 'opacity': 1}

    style_dict[str(i)] = inner_dict
districts_df = gdf_districts[['geometry']]

districts_gdf = gdp.GeoDataFrame(districts_df)

districts_gdf = districts_gdf.drop_duplicates().reset_index()

districts_gdf.head()
# Initialize a map

m_time_slider_AQI = folium.Map([47.4857,19.0902], tiles='Stamen Toner', zoom_start=11,min_zoom=2, max_bounds=True)



# Draw a TimeSliderChoropleth on it

g = TimeSliderChoropleth(

    districts_gdf.to_json(),

    styledict=style_dict



).add_to(m_time_slider_AQI)



# Add a GeoJsonTooltip to the map to show district number when hovering the mouse over it

geojson1 = folium.GeoJson(data=districts.to_json(),

                          style_function=lambda x: {'color':'white','fillColor':'transparent','weight':0.5},

                          tooltip=folium.GeoJsonTooltip(fields=['district_code'],

                                                        labels=False,

                                                        sticky=True),

                          highlight_function=lambda x: {'weight':2,'fillColor':'grey'}

                        

            ).add_to(m_time_slider_AQI)



# Show the scale of values

_ = cmap.add_to(m_time_slider_AQI)

cmap.caption = "LOG Air Quality Index - the max median among 5 parameters over 2016-2019"



# Save and display the map

m_time_slider_AQI.save(outfile='timeslider_budapest_AQI_2016-2019.html')

m_time_slider_AQI
median_AQI_per_param.head()
fg_df = median_AQI_per_param.reset_index()

fg_df.columns = ['station_in_district','parameter','value']



fg_df = closest_stations.merge(fg_df[['station_in_district','parameter','value']], left_on='closest_station', right_on='station_in_district')

fg_df.drop(['closest_station','station_in_district'], axis=1, inplace=True)



fg_df.set_index(['district_id'], inplace=True)



fg_df.head()
co = fg_df[fg_df['parameter'] == ' co']

no2 = fg_df[fg_df['parameter'] == 'no2']

o3 = fg_df[fg_df['parameter'] == 'o3']

pm10 = fg_df[fg_df['parameter'] == 'pm10']

so2 = fg_df[fg_df['parameter'] == 'so2']
# Initialize a map

m_parameters = folium.Map([47.4857,19.0902], tiles='cartodbpositron', zoom_start=11,min_zoom=2, max_bounds=True)



# Create feature groups

feature_group0 = folium.FeatureGroup(name='co',overlay=True).add_to(m_parameters)

feature_group1= folium.FeatureGroup(name='no2',overlay=True).add_to(m_parameters)

feature_group2 = folium.FeatureGroup(name='o3',overlay=True).add_to(m_parameters)

feature_group3= folium.FeatureGroup(name='pm10',overlay=True).add_to(m_parameters)

feature_group4 = folium.FeatureGroup(name='so2',overlay=True).add_to(m_parameters)



fs = [feature_group0,feature_group1,feature_group2,feature_group3,feature_group4]

parameters = [co.value,no2.value,o3.value,pm10.value,so2.value]



# Add a choropleth map for each parameter

for i in range(len(parameters)): 

    choropleth1 = folium.Choropleth(

    geo_data=districts.to_json(),

    name='choropleth',

    data=parameters[i],

    key_on='feature.id',

    fill_color='YlGn',

    nan_fill_color="black",

    fill_opacity=0.7,

    line_opacity=0.2,

    highlight=True,

    line_color='black').geojson.add_to(fs[i])



# Add a GeoJsonTooltip to the map to show district number when hovering the mouse over it

geojson1 = folium.GeoJson(data=districts.to_json(),

                          style_function=lambda x: {'color':'green','fillColor':'transparent','weight':0.5},

                          tooltip=folium.GeoJsonTooltip(fields=['district_code'],

                                                        labels=False,

                                                        sticky=True),

                          highlight_function=lambda x: {'weight':2,'fillColor':'grey'},

                        

                         ).add_to(choropleth1)

    

# Show the scale of values

colormap = linear.YlGn_09.scale(

fg_df.value.min(),

fg_df.value.max()).to_step(10)

colormap.caption = 'Median AQI for each parameter over 2016-2019'

colormap.add_to(m_parameters)

   

# Add layer control tool that lets you choose how many parameters to show on the map

folium.LayerControl(collapsed=False).add_to(m_parameters)



# Save and display the map

m_parameters.save('median_AQI_per_parameter.html')

m_parameters