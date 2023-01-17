import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import plotly.express as px

import plotly.graph_objects as go

import copy

import json

from urllib.request import urlopen

from plotly.subplots import make_subplots

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')

geo_china = pd.read_csv("../input/chinese-cities/china_coordinates.csv")

geo_world = pd.read_csv("../input/world-coordinates/world_coordinates.csv")

wikipedia_iso_country_codes = pd.read_csv("../input/countries-iso-codes/wikipedia-iso-country-codes.csv")
with urlopen('https://raw.githubusercontent.com/stjacob/china_geojson/master/china.geojson') as response:

    counties = json.load(response)
#counties['features'][0]
df['Country'] = df['Country'].replace('Mainland China', 'China')
df2 = copy.copy(df)

df = df.drop(['Sno', 'Date'], axis  = 1)

df['Last Update'] = pd.to_datetime(df['Last Update'])

df[['Province/State', 'Country']] = df[['Province/State', 'Country']].fillna('Unavailable')

df[['Confirmed', 'Deaths', 'Recovered']] = df[['Confirmed', 'Deaths', 'Recovered']].fillna(0.0)

#df.head(5)
latest_data_df = df.groupby(['Country', 'Province/State'])['Last Update', 'Confirmed', 'Deaths', 'Recovered'].max().reset_index()

latest_data_df = latest_data_df[['Country', 'Province/State', 'Confirmed', 'Recovered', 'Deaths', 'Last Update']]

#latest_data_df.shape
#latest_data_df.head(5)

#df[df['Country'] == 'Australia'].tail(5)

#latest_data_df.head(5)
china_df = latest_data_df[latest_data_df['Country'] == 'China'].reset_index(drop = True)

#china_df.head(5)

grouped_cnf_df = latest_data_df.groupby(['Country'])['Confirmed', 'Recovered', 'Deaths'].sum().reset_index()

#grouped_cnf_df.head(5)

grouped_cnf_df = grouped_cnf_df[(grouped_cnf_df['Country'] != 'China') & (grouped_cnf_df['Country'] != 'Others')]

#grouped_cnf_df.head(5)
fig = px.bar(grouped_cnf_df, x="Confirmed", y="Country", orientation='h', color = "Confirmed", height = 600)

fig.update_layout(yaxis={'categoryorder':'total ascending'}, title = 'CONFIRMED CASES in countries other than China')

fig.show()

figz = px.bar(df2[(df2['Country'] != 'China') & (df2['Country'] != 'Others')], x="Confirmed", y="Country", orientation = 'h',animation_frame="Date", hover_name="Country", range_x = [0,55], height = 800, color = 'Confirmed')

figz.update_layout(yaxis={'categoryorder':'total ascending'}, title = 'CONFIRMED CASES in countries other than China')

figz.show()
#df2.head(5)

#figz = px.bar(df2[(df2['Country'] != 'China') & (df2['Country'] != 'Others')], x="Country", y="Confirmed", animation_frame="Date", hover_name="Country", range_y = [0,60], color = 'Confirmed')

#figz.layout.updatemenus[0].buttons[0].args[1]["frame"]["duration"] = 1000

#figz.show()
fig = go.Figure(go.Bar(x=grouped_cnf_df['Confirmed'], y=grouped_cnf_df['Country'], name='Confirmed', orientation = 'h'))

fig.add_trace(go.Bar(x=grouped_cnf_df['Deaths'], y=grouped_cnf_df['Country'], name='Deaths', orientation = 'h'))

fig.add_trace(go.Bar(x=grouped_cnf_df['Recovered'], y=grouped_cnf_df['Country'], name='Recovered', orientation = 'h'))



fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'}, height = 1000, title = 'PATIENT DISTRIBUTION in countries other than China')

fig.show()
fig = go.Figure(go.Bar(x=china_df['Confirmed'], y=china_df['Province/State'], name='Confirmed', orientation = 'h'))

fig.add_trace(go.Bar(x=china_df['Deaths'], y=china_df['Province/State'], name='Deaths', orientation = 'h'))

fig.add_trace(go.Bar(x=china_df['Recovered'], y=china_df['Province/State'], name='Recovered', orientation = 'h'))



fig.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'}, height = 1000, title = 'PATIENT DISTRIBUTION in Chinese Provinces')

fig.show()
fig = make_subplots(rows=2, cols=1, start_cell="bottom-left", row_heights = [0.96, 0.04], vertical_spacing = .09)





fig.add_trace(go.Bar(x=china_df[china_df['Province/State'] != 'Hubei']['Confirmed'], y=china_df[china_df['Province/State'] != 'Hubei']['Province/State'], name='Confirmed', orientation = 'h'), row = 1, col=1)

fig.add_trace(go.Bar(x=china_df[china_df['Province/State'] != 'Hubei']['Deaths'], y=china_df[china_df['Province/State'] != 'Hubei']['Province/State'], name='Deaths', orientation = 'h'), row = 1, col=1)

fig.add_trace(go.Bar(x=china_df[china_df['Province/State'] != 'Hubei']['Recovered'], y=china_df[china_df['Province/State'] != 'Hubei']['Province/State'], name='Recovered', orientation = 'h'), row = 1, col=1)



fig.add_trace(go.Bar(x=china_df[china_df['Province/State'] == 'Hubei']['Confirmed'], y=china_df[china_df['Province/State'] == 'Hubei']['Province/State'], name='Confirmed', orientation = 'h'), row = 2, col=1)

fig.add_trace(go.Bar(x=china_df[china_df['Province/State'] == 'Hubei']['Deaths'], y=china_df[china_df['Province/State'] == 'Hubei']['Province/State'], name='Deaths', orientation = 'h'), row = 2, col=1)

fig.add_trace(go.Bar(x=china_df[china_df['Province/State'] == 'Hubei']['Recovered'], y=china_df[china_df['Province/State'] == 'Hubei']['Province/State'], name='Recovered', orientation = 'h'), row = 2, col=1)





fig.update_layout(showlegend=False, barmode='stack', yaxis={'categoryorder':'total ascending'}, height = 700, title = 'Wuhan (Hubei Province) versus other Chinese provinces')

fig.show()
china_df['Province/State'] = china_df['Province/State'].str.replace('Inner Mongolia', 'Nei Mongol')

china_df['Province/State'] = china_df['Province/State'].str.replace('Hong Kong', 'HongKong')

china_df['Province/State'] = china_df['Province/State'].str.replace('Xinjiang', 'Xinjiang Uygur')

china_df['Province/State'] = china_df['Province/State'].str.replace('Tibet', 'Xizang')

china_df['Province/State'] = china_df['Province/State'].str.replace('Ningxia', 'Ningxia Hui')

china_df = china_df.sort_values(['Province/State'])
def read_geojson(url):

    with urlopen(url) as url:

        jdata = json.loads(url.read().decode())

    if   'id'  not in jdata['features'][0].keys():

        if 'properties' in jdata['features'][0].keys():

            if 'id' in jdata['features'][0]['properties']  and jdata['features'][0]['properties']['id'] is not None:

                for k, feat in enumerate(jdata['features']):

                    jdata['features'][k]['id'] = feat['properties']['id']

            else:

                for k in range(len(jdata['features'])):

                    jdata['features'][k]['id'] = k

    return jdata



jdata = read_geojson('https://raw.githubusercontent.com/stjacob/china_geojson/master/china.geojson')
locations = [i for i in range(0,34)]

text = [feat['properties']['NAME_1']  for feat in jdata['features'] if feat['id'] in locations] #province names

china_geojson_df = pd.DataFrame(columns = ['Province/State'])

china_geojson_df['Province/State'] = text
china_df = china_df.reindex([0,1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,21,22,23,24,25,26,27,29,30,31,32,33,28,12,20]).reset_index(drop=True)

china_df = china_df.reindex([0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,18,14,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33]).reset_index(drop=True)

china_geojson_df = pd.merge(china_df, china_geojson_df, left_on='Province/State', right_on='Province/State')
mapbox_access_token = "pk.eyJ1IjoiYmF0byIsImEiOiJjamJwZzRvaGE2MTljMzJtcjhzaDJvaXFxIn0.TkTLg13Af-ERPjOWzB-BFQ"

trace = go.Choroplethmapbox(z=china_geojson_df['Confirmed'].tolist(),

                            locations=locations,

                            colorbar=dict(thickness=20, ticklen=3),

                            colorscale='Viridis',

                            geojson=jdata,

                            text=china_geojson_df['Province/State'].tolist(),

                            marker_line_width=0.1, marker_opacity=0.7)

                            

                            

layout = go.Layout(title_text= 'Confirmed Cases - Chinese Provinces',

                   mapbox = dict(center= dict(lat=39.913818,  lon=102.363625),

                                 accesstoken= mapbox_access_token,

                                 zoom=3.0,

                               ))



fig = go.Figure(data=[trace], layout =layout)

fig.update_layout(mapbox_style = "carto-positron", height = 700, width = 1100)

fig.show()
jdata_world = read_geojson('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')

world_locations = [i for i in range(0,180)]

country_names = [feat['properties']['name']  for feat in jdata_world['features']] #country names
grouped_cnf_df['Country'] = grouped_cnf_df['Country'].str.replace('US', 'United States of America')

grouped_cnf_df['Country'] = grouped_cnf_df['Country'].str.replace('UK', 'United Kingdom')

#grouped_cnf_df['Country'].tolist()

grouped_cnf_df = pd.merge(grouped_cnf_df, wikipedia_iso_country_codes, how='right', left_on='Country', right_on='English short name lower case')

grouped_cnf_df = grouped_cnf_df[['Country', 'Confirmed', 'Recovered', 'Deaths', 'Alpha-3 code']]

grouped_cnf_df = grouped_cnf_df.dropna()
for col in grouped_cnf_df.columns:

    grouped_cnf_df[col] = grouped_cnf_df[col].astype(str)

    

def get_text(row):

     return row['Country'] + '<br>' + 'Confirmed: ' + row['Confirmed'] + '<br>' + 'Recovered: ' + row['Recovered'] + '<br>' + 'Deaths: ' + row['Deaths']



grouped_cnf_df['text'] = grouped_cnf_df.apply(lambda row: get_text(row), axis = 1)
fig = px.choropleth(grouped_cnf_df, locations="Alpha-3 code",

                    color="Confirmed", # lifeExp is a column of gapminder

                    hover_name='text', # column to add to hover information

                    color_continuous_scale='Viridis')

                    #color_continuous_scale='Inferno')



fig.update_layout(title_text = 'Confirmed Cases - World')





fig.show()