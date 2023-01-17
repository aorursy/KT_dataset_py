import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import folium
import json

df_police_killing = pd.read_csv('../input/police-violence-in-the-us/police_killings.csv')
df_police_killing.dropna(axis=1, how='all', inplace=True)
usa_population = pd.read_csv('../input/usa-states-geojson/usa_population_2019.csv')
state_geo = '../input/usa-states-geojson/us-states.json'
import seaborn as sns; sns.set()
# it seems some columns are null
df_police_killing.head()
df_police_killing.dropna(axis=0,how='all',inplace=True)
df_police_killing.shape
killing_per_state = df_police_killing[['State','ID']].groupby(['State']).count().reset_index()
killing_per_state.columns = ['State', 'Kills']
killing_per_state.head()
map_usa = folium.Map(location=[37, -102], zoom_start=5)

folium.Choropleth(
    state_geo,
    name='choropleth',
    data=killing_per_state,
    columns=['State', 'Kills'],
    key_on='feature.id',
    fill_color='YlGn',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of police Kills by state',
    highlight=True
).add_to(map_usa)

map_usa
usa_population.tail()
kill_and_pop_per_state = killing_per_state.merge(usa_population,how='inner',left_on='State',right_on='Postal Code')
kill_and_pop_per_state.head()
kill_and_pop_per_state['Killings per 1M ppl'] = kill_and_pop_per_state['Kills']/kill_and_pop_per_state['Total Resident Population']*1000000
kill_and_pop_per_state.head()
# Add properties to geojson, in order to add tooltips
import json
with open(state_geo, encoding="utf8") as f:
    map_data = json.load(f)

# properties 
[key for key in map_data['features'][0]['properties']]
states_order = [state['id'] for state in map_data['features']]
for idx in range(len(states_order)):
    map_data['features'][idx]['properties']['Killings per 1M ppl'] = round(kill_and_pop_per_state[kill_and_pop_per_state.State == states_order[idx]]['Killings per 1M ppl'].values[0],2)
    map_data['features'][idx]['properties']['Kills'] = int(kill_and_pop_per_state[kill_and_pop_per_state.State == states_order[idx]]['Kills'].values[0])
    map_data['features'][idx]['properties']['Population'] = int(kill_and_pop_per_state[kill_and_pop_per_state.State == states_order[idx]]['Total Resident Population'].values[0])
map_usa = folium.Map(location=[37, -102], zoom_start=5)

chorop_map = folium.Choropleth(
    geo_data=map_data,
    name='choropleth',
    data=kill_and_pop_per_state,
    columns=['State', 'Killings per 1M ppl'],
    key_on='feature.id',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of police kills every 1M ppl by state',
    highlight=True
).add_to(map_usa)

folium.LayerControl().add_to(map_usa)
chorop_map.geojson.add_child(
    folium.features.GeoJsonTooltip(['name', 'Killings per 1M ppl', 'Kills','Population'])
)

map_usa
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
import folium.plugins as plugins
df_police_killing['Address'] = df_police_killing["Street Address of Incident"] + "," + df_police_killing["City"] + "," + df_police_killing["State"]
df_police_killing.Address
locator = Nominatim(user_agent="myGeocoder")
location = locator.geocode("Champ de Mars, Paris, France")
print("Latitude = {}, Longitude = {}".format(location.latitude, location.longitude))
df_police_killing_test = df_police_killing.iloc[:100].copy()
## LONG RUNNING TIME: ~10:43 to
# 1 - conveneint function to delay between geocoding calls
geocode = RateLimiter(locator.geocode, min_delay_seconds=1/20)
# 2- - create location column
df_police_killing_test['location'] = df_police_killing_test['Address'].apply(geocode)
# 3 - create longitude, latitude and altitude from location column (returns tuple)
df_police_killing_test['point'] = df_police_killing_test['location'].apply(lambda loc: tuple(loc.point) if loc else None)
# 4 - split point column into latitude, longitude and altitude columns
df_police_killing_test['latitude'] = df_police_killing_test['point'].apply(lambda loc: None if loc is None else loc[0])
df_police_killing_test['longitude'] = df_police_killing_test['point'].apply(lambda loc: None if loc is None else loc[1])
df_police_killing_test = df_police_killing_test[pd.notnull(df_police_killing_test.latitude)].copy()
map_usa = folium.Map(location=[37, -102], zoom_start=5)

chorop_map = folium.Choropleth(
    geo_data=map_data,
    name='choropleth',
    data=kill_and_pop_per_state,
    columns=['State', 'Killings per 1M ppl'],
    key_on='feature.id',
    fill_color='YlOrRd',
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name='Number of police kills every 1M ppl by state',
    highlight=True
).add_to(map_usa)

folium.LayerControl().add_to(map_usa)
chorop_map.geojson.add_child(
    folium.features.GeoJsonTooltip(['name', 'Killings per 1M ppl', 'Kills','Population'])
)


plugins.FastMarkerCluster(data=list(zip(df_police_killing_test['latitude'].values, df_police_killing_test['longitude'].values))).add_to(chorop_map)
folium.LayerControl().add_to(chorop_map)


map_usa