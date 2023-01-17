import numpy as np 

import pandas as pd

import os

import matplotlib.pyplot as plt

import seaborn as sns 

import datetime as dt

%matplotlib inline

import datetime as dt

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly import tools

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

from shapely.geometry import shape, Point, Polygon

import folium

from folium.plugins import HeatMap, HeatMapWithTime

init_notebook_mode(connected=True)
data_df = pd.read_csv("/kaggle/input/forest-surfaces-evolution-in-romania-19902019/forest_surfaces_romania_en_1990_2019.csv")
data_df.shape
data_df.head()
data_df.info()
data_df.Category.unique()
print(list(data_df.Region.unique()))
total = data_df.loc[data_df.Region=='TOTAL']

totalT = total.loc[total.Category=='Total']

totalH = total.loc[total.Category=='Hardwood']

totalS = total.loc[total.Category=='Softwood']

totalO = total.loc[total.Category=='Others']



traceT = go.Scatter(

    x = totalT['Year'],y = totalT['Value'],

    name="Total surface",

    marker=dict(color="Green"),

    mode = "markers+lines",

    text=totalT['Value']

)



traceH = go.Scatter(

    x = totalH['Year'],y = totalH['Value'],

    name="Hardwood",

    marker=dict(color="Lightgreen"),

    mode = "markers+lines",

    text=totalH['Value']

)



traceS = go.Scatter(

    x = totalS['Year'],y = totalS['Value'],

    name="Softwood",

    marker=dict(color="Darkgreen"),

    mode = "markers+lines",

    text=totalS['Value']

)



traceO = go.Scatter(

    x = totalO['Year'],y = totalO['Value'],

    name="Other surfaces",

    marker=dict(color="Brown"),

    mode = "markers+lines",

    text=totalO['Value']

)



data = [traceT, traceH, traceS, traceO]



layout = dict(title = 'Total surfaces, Hardwood, Softwood and other surfaces - forest Romania 1990-2019',

          xaxis = dict(title = 'Year', showticklabels=True), 

          yaxis = dict(title = 'Surface [Thousands hectares]'),

          hovermode = 'closest'

         )

fig = dict(data=data, layout=layout)

iplot(fig, filename='forest-surface')
county_list = ['Bihor', 'Bistrita-Nasaud', 'Cluj', 'Maramures', 'Satu Mare', 'Salaj', 'Alba', 'Brasov', 'Covasna', 'Harghita', 'Mures', 'Sibiu', 

               'Bacau', 'Botosani', 'Iasi', 'Neamt', 'Suceava', 'Vaslui', 'Braila', 'Buzau', 'Constanta', 'Galati', 'Tulcea', 'Vrancea', 

               'Arges', 'Calarasi', 'Dambovita', 'Giurgiu', 'Ialomita', 'Prahova', 'Teleorman', 'Ilfov', 'Bucuresti', 

               'Dolj', 'Gorj', 'Mehedinti', 'Olt', 'Valcea', 'Arad', 'Caras-Severin', 'Hunedoara', 'Timis']

data_county_df = data_df.loc[data_df.Region.isin(county_list)]
selection_df = data_county_df.loc[data_county_df.Category=="Total"]

min_value = selection_df.Value.min()

max_value = selection_df.Value.max()

import plotly.express as px

fig = px.bar(selection_df, x="Value", y="Region", animation_frame='Year', orientation='h',

             range_color =[min_value,max_value],

             width=600, height=800, range_x = [min_value,max_value],

            title='Surface evolution/county 1990-2019 - Total')

fig.update_layout(font=dict(family="Courier New, monospace",size=10,color="#7f7f7f"))

fig.show()
selection_df = data_county_df.loc[data_county_df.Category=="Hardwood"]

min_value = selection_df.Value.min()

max_value = selection_df.Value.max()

import plotly.express as px

fig = px.bar(selection_df, x="Value", y="Region", animation_frame='Year', orientation='h',

             range_color =[min_value,max_value],

             width=600, height=800, range_x = [min_value,max_value],

            title='Surface evolution/county 1990-2019 - Hardwood')

fig.update_layout(font=dict(family="Courier New, monospace",size=10,color="#7f7f7f"))

fig.show()
selection_df = data_county_df.loc[data_county_df.Category=="Softwood"]

min_value = selection_df.Value.min()

max_value = selection_df.Value.max()

import plotly.express as px

fig = px.bar(selection_df, x="Value", y="Region", animation_frame='Year', orientation='h',

             range_color =[min_value,max_value],

             width=600, height=800, range_x = [min_value,max_value],

            title='Surface evolution/county 1990-2019 - Softwood')

fig.update_layout(font=dict(family="Courier New, monospace",size=10,color="#7f7f7f"))

fig.show()
selection_df = data_county_df.loc[data_county_df.Category=="Others"]

min_value = selection_df.Value.min()

max_value = selection_df.Value.max()

import plotly.express as px

fig = px.bar(selection_df, x="Value", y="Region", animation_frame='Year', orientation='h',

             range_color =[min_value,max_value],

             width=600, height=800, range_x = [min_value,max_value],

            title='Surface evolution/county 1990-2019 - Other surfaces')

fig.update_layout(font=dict(family="Courier New, monospace",size=10,color="#7f7f7f"))

fig.show()
ro_geo_data = "/kaggle/input/elementary-school-admission-romania-2014/romania.geojson"

ro_large_geo_data = "/kaggle/input/elementary-school-admission-romania-2014/ro_judete_poligon.geojson"
import difflib 

import json



# retrieve the county names from geoJson

with open(ro_geo_data) as json_file:

    json_data = json.load(json_file)

county_lat_long_df = pd.DataFrame()

for item in json_data['features']:

    polygons = list(shape(item['geometry']))

    county = item['properties']['name']

    county_lat_long_df = county_lat_long_df.append(pd.DataFrame({'Region': county, 'Lat':polygons[0].centroid.y, 'Long': polygons[0].centroid.x}, index=[0]))



# merge county data    

county_join = pd.DataFrame(data_county_df.Region.unique())

county_join.columns = ['Region']

# match the county names

difflib.get_close_matches

county_lat_long_df['Region'] = county_lat_long_df.Region.map(lambda x: difflib.get_close_matches(x, county_join.Region)[0])

print(f"Validation [polygons]: {county_lat_long_df.Region.nunique()},{county_lat_long_df.Region.nunique()}")



with open(ro_large_geo_data) as json_file:

    json_data = json.load(json_file)

county_population_df = pd.DataFrame()

for item in json_data['features']:

    county = item['properties']['name']

    population = item['properties']['pop2011']

    county_population_df = county_population_df.append(pd.DataFrame({'Region': county, 'population': population}, index=[0]))

difflib.get_close_matches

county_population_df['Region'] = county_population_df.Region.map(lambda x: difflib.get_close_matches(x, county_join.Region)[0])

print(f"Validation [population]: {county_population_df.Region.nunique()},{county_population_df.Region.nunique()}")
data_county_df = data_county_df.merge(county_lat_long_df, on="Region")
def plot_map(data_county_df, year='1990', category='Total', bins_max=500):

    

    last_data_df = data_county_df.loc[(data_county_df.Year == year) & (data_county_df.Category==category)].reset_index()

    

    ro_map = folium.Map(location=[45.9, 24.9], zoom_start=6)



    bins_list = []

    for i in range(0,9):

        bins_list.append(int(i * bins_max / 8))

   

    folium.Choropleth(

        geo_data=ro_geo_data,

        name='Counties countour plots',

        data=last_data_df,

        columns=['Region', 'Value'],

        key_on='feature.properties.name',

        fill_color='Greens',

        bins=bins_list,

        fill_opacity=0.6,

        line_opacity=0.5,

        legend_name=f'{category} forest surface [10^3 ha] / county ({year})'

    ).add_to(ro_map)







    radius_min = 3

    radius_max = 20

    weight = 1

    fill_opacity = 0.5



    _color_conf = 'Darkgreen'

    group0 = folium.FeatureGroup(name='<span style=\\"color: #20FF50;\\">Popups</span>')

    for i in range(len(last_data_df)):

        lat = last_data_df.loc[i, 'Lat']

        lon = last_data_df.loc[i, 'Long']

        _county = last_data_df.loc[i, 'Region']



        _radius_conf = np.sqrt(last_data_df.loc[i, 'Value'])

        if _radius_conf < radius_min:

            _radius_conf = radius_min



        if _radius_conf > radius_max:

            _radius_conf = radius_max



        _popup_conf = str(_county) + '\nSurface: '+str(last_data_df.loc[i, 'Value']) + ' [10^3 ha]'



        folium.CircleMarker(location = [lat,lon], 

                            radius = _radius_conf, 

                            popup = _popup_conf, 

                            tooltip = _popup_conf,

                            color = _color_conf, 

                            fill_opacity = fill_opacity,

                            weight = weight, 

                            fill = True, 

                            fillColor = _color_conf).add_to(group0)



    group0.add_to(ro_map)

    folium.LayerControl().add_to(ro_map)



    return ro_map
ro_map = plot_map(data_county_df, 1990, 'Total')

ro_map
ro_map = plot_map(data_county_df, 2019, 'Total')

ro_map
ro_map = plot_map(data_county_df, 1990, 'Hardwood',bins_max=400)

ro_map
ro_map = plot_map(data_county_df, 2019, 'Hardwood', bins_max=400)

ro_map
ro_map = plot_map(data_county_df, 1990, 'Softwood', bins_max=350)

ro_map
ro_map = plot_map(data_county_df, 2019, 'Softwood',bins_max=350)

ro_map
ro_map = plot_map(data_county_df, 1990, 'Others',bins_max=15)

ro_map
ro_map = plot_map(data_county_df, 2019, 'Others',bins_max=15)

ro_map