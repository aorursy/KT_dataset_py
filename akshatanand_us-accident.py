import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
data = pd.read_csv('../input/us-accidents/US_Accidents_Dec19.csv')
data.info()
data.shape
data.head(3).T
!pip install folium

import folium
def plot_map1(LatLong, city=None):
    accident_map = folium.Map(location=LatLong, 
                           tiles = "Stamen Toner",
                           zoom_start = 10)
    if city != None:
        data_heatmap = data[data["City"] == city]
    else:
        data_heatmap = data.copy()
    data_heatmap = data_heatmap[['Start_Lat','Start_Lng']]
    data_heatmap = [[row['Start_Lat'],row['Start_Lng']] for index, row in data_heatmap.iterrows()]
    HeatMap(data_heatmap, radius=10).add_to(accident_map)
    return accident_map
data[data['City']=='New York'].shape
from folium.plugins import HeatMap
plot_map1([40.712776,-74.005974], city='New York')
import plotly.graph_objects as go
state_count_acc = pd.value_counts(data['State'])

fig = go.Figure(data=go.Choropleth(
    locations=state_count_acc.index,
    z = state_count_acc.values.astype(float),
    locationmode = 'USA-states',
    colorscale = 'Reds',
    colorbar_title = "Count Accidents",
))

fig.update_layout(
    title_text = '2016 - 2019 US Traffic Accident Dataset by State',
    geo_scope='usa',
)

fig.show()
!pip install plotly-geo
df_county = pd.read_csv('https://raw.githubusercontent.com/plotly/datasets/master/laucnty16.csv')
df_county.head(3)
df_county['county_full'] = df_county['County Name/State Abbreviation'].apply(lambda x: x.split(', ')[0])
df_county['county_name'] = df_county['county_full'].apply(lambda x: x.split(' County')[0])

county_count_acc = pd.value_counts(data['County'])
fips_county_df = df_county[['county_name', 'County FIPS Code', 'State FIPS Code']].merge(county_count_acc, left_on='county_name', right_index=True)
import plotly.figure_factory as ff

fips_county_df['State FIPS Code'] = fips_county_df['State FIPS Code'].apply(lambda x: str(x).zfill(2))
fips_county_df['County FIPS Code'] = fips_county_df['County FIPS Code'].apply(lambda x: str(x).zfill(3))
fips_county_df['FIPS'] = fips_county_df['State FIPS Code'] + fips_county_df['County FIPS Code']

colorscale = ["#f7fbff", "#ebf3fb", "#deebf7", "#d2e3f3", "#c6dbef", "#b3d2e9", "#9ecae1",
    "#85bcdb", "#6baed6", "#57a0ce", "#4292c6", "#3082be", "#2171b5", "#1361a9",
    "#08519c", "#0b4083", "#08306b"
]
endpts = list(np.linspace(1,30000, len(colorscale) - 1))
fips = fips_county_df['FIPS'].tolist()
values = fips_county_df['County'].tolist()


fig = ff.create_choropleth(
    fips=fips, values=values, scope=['usa'],
    binning_endpoints=endpts, colorscale=colorscale,
    show_state_data=False,
    show_hover=True,
    asp = 2.9,
    title_text = 'USA County accidents count',
    legend_title = 'Accidents count'
)
fig.layout.template = None
fig.show()
data_sever = data.sample(n=10000)

fig = go.Figure(data=go.Scattergeo(
        locationmode = 'USA-states',
        lon = data_sever['Start_Lng'],
        lat = data_sever['Start_Lat'],
        text = data_sever['City'],
        mode = 'markers',
        marker = dict(
            size = 8,
            opacity = 0.8,
            reversescale = True,
            autocolorscale = False,
            symbol = 'circle',
            line = dict(
                width=1,
                color='rgba(102, 102, 102)'
            ),
            colorscale = 'Reds',
            cmin = data_sever['Severity'].max(),
        color = data_sever['Severity'],
        cmax = 1,
            colorbar_title="Severity"
        )))

fig.update_layout(
        title = 'Severity of accidents',
        geo = dict(
            scope='usa',
            projection_type='albers usa',
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.7,
            subunitwidth = 0.7
        ),
    )
fig.show()
