%%capture
!pip install census
!pip install us
# standard libraries
from datetime import datetime
import json
import os
import requests

# 3rd party librares
import branca.colormap as cm
from census import Census
from IPython.display import display
import folium
import geopandas as gpd
import pandas as pd
from shapely.geometry import Polygon
from us import states

# local
from kaggle_secrets import UserSecretsClient
# secrets
user_secrets = UserSecretsClient()
CENSUS_KEY = user_secrets.get_secret("CENSUS_KEY")
RAPID_API_KEY = user_secrets.get_secret("RAPID_API_KEY")

# RapidAPI Request Headers (kevincal unlimited key)
API_URL = "https://idealspot-geodata.p.rapidapi.com"

API_REQUEST_HEADERS = {
    'x-rapidapi-host': "idealspot-geodata.p.rapidapi.com",
    'x-rapidapi-key': RAPID_API_KEY
    }
# IdealSpot
url = API_URL + "/api/v1/geometries/geometry"

# Build `Location` Object to Query API
location = {
    "type":"region",
    "regiontype": "county",
    "region_id": "TravisCountyTX"
}

# Fetch Geometry Record from IdealSpot API
params = { "location": json.dumps(location) }
r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)
data = r.json().get('data')
display(data)
# looking at the `Feature Collection` above, get FIPS for Travis County, TX

# Get County FIPS code
travis_fips = data['features'][0]['properties']['_properties']['COUNTYFP']

# using FIPS, get the list of CENSUS County Tracts 
c = Census(CENSUS_KEY)
county_tracts = c.sf1.state_county_tract('NAME', states.TX.fips, travis_fips, Census.ALL)

# display
print("%s county tracts in Travis County, TX" % len(county_tracts))
display(county_tracts[:2])

# get home-value for East Austin
url = "https://idealspot-geodata.p.rapidapi.com/api/v1/data/insights/home-value"

# fetch API endpoint and display results
params = {
    "periods": "true",
    "parameter_options": "true",
    "parameters": "true"
    }

r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)
display(r.json())
# tip: keep a list of processed id's so no need to requery
processed_region_ids = []

# list of county tract data
county_tract_data = []
# iterate through county tracts to fetch geo and insight data
for ct in county_tracts:
    
    # debug 
    # print("Processing %s" % ct)
    
    # get tract FIPS
    name = ct.get('NAME')
    state_fips = ct.get('state')
    county_fips = ct.get('county')
    tract_fips = ct.get('tract')
    
    # build region_id
    region_id = "%s%s%s" % (state_fips, county_fips, tract_fips)
    
    # get duplicate
    if region_id in processed_region_ids:
        print("Duplicate. Skipping")
        continue
    else:
        processed_region_ids.append(region_id)
    
    # build location parameter for API Query
    location = {
        "type":"region", 
        "regiontype": "tract", 
        "region_id": region_id
        }
    
    # API params, using 
    params = {
        "version": "v2",
        "location": json.dumps(location)
    }
    
    # build Insight API URL
    url = API_URL + "/api/v1/data/insights/home-value/query"
    r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)
    insight_raw_data = r.json()
    insight_data = insight_raw_data.get('data')[0].get('data')
    median_home_value = None
    for (l, v) in insight_data:
        if l == 'Median Home Value':
            median_home_value = v
            
    # get Geometry API URL
    url = API_URL + "/api/v1/geometries/geometry"
    r = requests.get(url=url, params=params, headers=API_REQUEST_HEADERS)    
    geometry_raw_data = r.json()
    
    # get the polygon coordinates
    geometry = geometry_raw_data.get('data', {})\
        .get('features', [])[0].get('geometry')
    coordinates = geometry['coordinates'][0]
    poly = Polygon(coordinates)
    
    # build list
    county_tract_data.append({
        'region_name': name,
        'region_id': int(region_id),
        'Median Home Value': median_home_value,
        'geometry': poly
    })

# create GeoDataFrame
county_tracts_gdf = gpd.GeoDataFrame(county_tract_data, crs="EPSG:4326")
county_tracts_gdf.set_index('region_id')
county_tracts_gdf.head()
# center map based on features
lng_map = county_tracts_gdf.centroid.x.mean()
lat_map = county_tracts_gdf.centroid.y.mean()

# create folium map
map = folium.Map(
    location=[lat_map, lng_map],
    zoom_start=11,
    tiles=None)

# set tilelayer manually for more control
tile_layer = folium.TileLayer('CartoDB positron', name="Light Map", control=False).add_to(map)

# build color scale
threshold_scale = (county_tracts_gdf['Median Home Value']
                       .quantile((0, 0.1, 0.75, 0.9, 0.98, 1))).tolist()

colormap = cm.linear.YlGnBu_09.to_step(
    data=county_tracts_gdf['Median Home Value'], 
    method='quant', 
    quantiles=[0, 0.1, 0.75, 0.9, 0.98,1 ])

# build choropleth
folium.Choropleth(
     geo_data=county_tracts_gdf,
     name='Median Home Value',
     data=county_tracts_gdf,
     columns=['region_id', 'Median Home Value'],
     key_on="feature.properties.region_id",
     fill_color='YlGnBu',
     threshold_scale=threshold_scale,
     fill_opacity=0.6,
     line_opacity=0.2,
     legend_name='Median Home Value',
     smooth_factor=0
    ).add_to(map)

# display the static map
display(map)

# build interactive tooltip
style_function = lambda x: {
    'weight': 0.1, 
    'color': 'black',
    'fillColor': colormap(x['properties']['Median Home Value']), 
    'fillOpacity': 0.01
    }

highlight_function = lambda x: {
    'fillColor': '#000000', 
    'color':'#000000', 
    'fillOpacity': 0.2, 
    'weight': 0.1
    }

tooltip_overlay=folium.features.GeoJson(
        county_tracts_gdf,
        style_function=style_function,
        control=False,
        highlight_function=highlight_function,
        tooltip=folium.features.GeoJsonTooltip(
            fields=['region_name', 'Median Home Value'],
            aliases=['County Tract', 'Median Home Value ($)'],
            style=("background-color: white; color: #333333; font-family: arial; font-size: 12px; padding: 10px;"),
            sticky=True,
            localize=True
        )
    )
map.add_child(tooltip_overlay)
map.keep_in_front(tooltip_overlay)

# add layer control
folium.LayerControl().add_to(map)

# display map
display(map)
