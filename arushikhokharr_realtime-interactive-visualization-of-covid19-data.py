import requests

import pandas as pd
raw= requests.get("https://services1.arcgis.com/0MSEUqKaxRlEPj5g/arcgis/rest/services/ncov_cases2_v1/FeatureServer/2/query?where=1%3D1&outFields=*&outSR=4326&f=json")

raw_json = raw.json()

d = pd.DataFrame(raw_json["features"])

d.head(2)
data_list = d["attributes"].tolist()

df = pd.DataFrame(data_list)

df.set_index("OBJECTID")

df = df[["Country_Region", "Confirmed", "Deaths", "Recovered"]]

df.head()
import geopandas as gpd
mapp=gpd.read_file('../input/shapefile/custom.geo.json')

mapp.head()
mapp = mapp.rename(columns={'geometry': 'geometry','name':'Country_Region'}).set_geometry('geometry')

mapp.drop(mapp.columns.difference(['Country_Region','geometry']), 1, inplace=True)

mapp["Country_Region"].replace({"United States": "US", "Korea": "Korea, South", "Macedonia": "North Macedonia","Dominican Rep.":"Dominican Republic","Eq. Guinea":"Equatorial Guinea","Bosnia and Herz.":"Bosnia and Herzegovina","Czech Rep.":"Czechia","Dem. Rep. Congo":"Congo (Kinshasa)","Lao PDR":"Laos","Taiwan":"Taiwan*","Central African Rep.":"Central African Republic","W. Sahara":"Western Sahara","Greenland":"Denmark","Côte d'Ivoire":"Cote d'Ivoire","Congo":"Congo (Brazzaville)","S. Sudan":"South Sudan"}, inplace=True)

mapp.head(2)
merged = mapp.merge(df, on = 'Country_Region', how='left')

merged[5:9]
import json

merged_json = json.loads(merged.to_json())

json_data = json.dumps(merged_json)
from bokeh.io import output_notebook, show, output_file

from bokeh.plotting import figure

from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar, HoverTool

from bokeh.palettes import brewer
geosource = GeoJSONDataSource(geojson = json_data)

palette = brewer["Blues"][9]

palette = palette[::-1]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.

color_mapper = LinearColorMapper(palette = palette, low = -500000, high = 2000000)

#Create figure object.

p = figure(title = '', plot_height = 400 , plot_width = 810, toolbar_location = None)

p.xgrid.grid_line_color = None

p.ygrid.grid_line_color = None

p.xaxis.visible = False

p.yaxis.visible = False

p.background_fill_color = "#D3D3D3"

#Add patch renderer to figure. 

p.patches('xs','ys', source = geosource,fill_color = {'field' :'Confirmed', 'transform' : color_mapper},

          line_color = '#D3D3D3', line_width = 0.6, fill_alpha = 1)

hover = HoverTool(tooltips = [ ('Country: ','@Country_Region'),('Confirmed: ', '@Confirmed'),('Deaths: ','@Deaths')])

p.add_tools(hover)

#Display figure inline in notebook.

output_notebook()

#Display figure.

show(p)
import plotly.graph_objects as go

import plotly.express as px

from plotly.subplots import make_subplots
df_top10 = df.nlargest(10, "Confirmed")

top10_countries_1 = df_top10["Country_Region"].tolist()

top10_confirmed = df_top10["Confirmed"].tolist()



df_top20 = df.nlargest(10, "Recovered")

top10_countries_2 = df_top20["Country_Region"].tolist()

top10_recovered = df_top20["Recovered"].tolist()



df_top30 = df.nlargest(10, "Deaths")

top10_countries_3 = df_top30["Country_Region"].tolist()

top10_deaths = df_top30["Deaths"].tolist()
total_confirmed = df["Confirmed"].sum()

total_recovered = df["Recovered"].sum()

total_deaths = df["Deaths"].sum()
fig = make_subplots(

    rows = 4, cols = 3,



    specs=[

            [    {"type": "indicator"}, {"type": "indicator"}, {"type": "indicator"} ],

            [    {"type": "bar", "colspan":3}, None, None],

            [    {"type": "bar", "colspan":3}, None, None],

            [    {"type": "bar", "colspan":3}, None, None],

          ]

)
fig.add_trace(

    go.Indicator(

        mode="number",

        value=total_confirmed,

        title="Confirmed Cases",

    ),

    row=1, col=1

)

fig.add_trace(

    go.Indicator(

        mode="number",

        value=total_recovered,

        title="Recovered Cases",

    ),

    row=1, col=2

)



fig.add_trace(

    go.Indicator(

        mode="number",

        value=total_deaths,

        title="Deaths",

    ),

    row=1, col=3

)



fig.add_trace(

    go.Bar(

        x=top10_countries_1,

        y=top10_confirmed, 

        name= "Confirmed Cases",

        marker=dict(color="#001a66"), 

        showlegend=True,

    ),

    row=2, col=1

)



fig.add_trace(

    go.Bar(x=top10_countries_2,

        y=top10_recovered, 

        name= "Recovered Cases",

        marker=dict(color="#004d99"), 

        showlegend=True),

    row=3, col=1

)



fig.add_trace(

    go.Bar(

        x=top10_countries_3,

        y=top10_deaths, 

        name= "Deaths",

        marker=dict(color="#809fff"), 

        showlegend=True),

    row=4, col=1

)

fig.update_layout(

    template="plotly_white",

    title = "Global COVID-19 Cases",

    showlegend=True,

    legend_orientation="h",

    legend=dict(x=0.65, y=0.8),

)
