#Processing

import numpy as np # linear algebra

import pandas as pd # data analysis , CSV file I/O (e.g. pd.read_csv)



#Visualization

import seaborn as sns

import plotly.express as px # plotting library

import matplotlib.pyplot as plt # plotting library

import folium # plotting library

import folium.plugins as plugins

import geopandas as gpd

from folium.plugins import TimestampedGeoJson

%matplotlib inline 



import plotly.express as px

import plotly.graph_objects as go



from sklearn.cluster import KMeans 

from sklearn.datasets.samples_generator import make_blobs



print('Libraries imported.')



# tranforming json file into a pandas dataframe library

from pandas.io.json import json_normalize



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
# Loading files from kaggle Database

import pandas as pd

COVID_confirmed = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv')

COVID_death = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv')

COVID_recovered = pd.read_csv('../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv')

COVID_India = pd.read_csv("../input/covid19-corona-virus-india-dataset/complete.csv")
# Loading files from kaggle Database

COVID_India.rename(columns={'Name of State / UT':'State/UnionTerritory','Cured/Discharged/Migrated':'Cured'},inplace=True)

COVID_India['Date']=COVID_India['Date'].apply(lambda x:x.replace('-','/'))

Latest_date=list(COVID_India['Date'])[-1]
COVID_confirmed.info()
COVID_confirmed['3/23/20'].fillna(0,inplace=True)

COVID_confirmed['3/23/20'].astype('int64')
COVID_confirmed.head()
COVID_India.head()
#Transposing Data for confirmed cases

COVID_confirmed_T=COVID_confirmed.T

COVID_confirmed_T.drop(index=['Lat','Long','Province/State'],inplace=True)

COVID_confirmed_T.columns=COVID_confirmed_T.iloc[0]

COVID_confirmed_T=COVID_confirmed_T[1:]

COVID_confirmed_T.set_index(COVID_confirmed_T.index.map(lambda x: pd.to_datetime(x, errors='ignore')),inplace=True)

COVID_confirmed_T['Global']=COVID_confirmed.sum()[2:]

COVID_confirmed_T.head()



#Transposing Data for Death

COVID_death_T=COVID_death.T

COVID_death_T.drop(index=['Lat','Long','Province/State'],inplace=True)

COVID_death_T.columns=COVID_death_T.iloc[0]

COVID_death_T=COVID_death_T[1:]

COVID_death_T['Global']=COVID_death.sum()[2:]

COVID_death_T.set_index(COVID_death_T.index.map(lambda x: pd.to_datetime(x, errors='ignore')),inplace=True)



#Transposing Data for Recovered

COVID_recovered_T=COVID_recovered.T

COVID_recovered_T.drop(index=['Lat','Long','Province/State'],inplace=True)

COVID_recovered_T.columns=COVID_recovered_T.iloc[0]

COVID_recovered_T=COVID_recovered_T[1:]

COVID_recovered_T['Global']=COVID_recovered.sum()[2:]

COVID_recovered_T.set_index(COVID_recovered_T.index.map(lambda x: pd.to_datetime(x, errors='ignore')),inplace=True)
fig = go.Figure()

fig.add_trace(

    go.Scatter(x=COVID_confirmed_T.index, y=COVID_confirmed.sum()[2:],name='Confirmed: Global',line=dict(color='blue')))

fig.add_trace(

    go.Scatter(x=COVID_death_T.index, y=COVID_death.sum()[2:],name='Death: Global',line=dict(color='red')))

fig.add_trace(

    go.Scatter(x=COVID_recovered_T.index, y=COVID_recovered.sum()[2:],name='Recovered: Global',line=dict(color='green')))

fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(

    go.Scatter(x=COVID_confirmed_T.index, y=COVID_confirmed_T['India'],name='Confirmed: India',line=dict(color='blue')))

fig.add_trace(

    go.Scatter(x=COVID_death_T.index, y=COVID_death_T['India'],name='Death: India',line=dict(color='red')))

fig.add_trace(

    go.Scatter(x=COVID_recovered_T.index, y=COVID_recovered_T['India'],name='Recovered: India',line=dict(color='green')))

fig.show()
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(

    go.Bar(x=COVID_confirmed_T.index, y=COVID_confirmed_T['India'],name='Confirmed: India'))

fig.add_trace(

    go.Bar(x=COVID_confirmed_T.index, y=COVID_confirmed_T['Italy'],name='Confirmed: Italy'))

fig.add_trace(

    go.Bar(x=COVID_confirmed_T.index, y=COVID_confirmed_T['Spain'],name='Confirmed: Spain'))

fig.show()
COVID_India
COVID_India_=COVID_India[COVID_India['Date']==Latest_date].sort_values(by='Total Confirmed cases')
import plotly.express as px

import plotly.graph_objects as go



fig = go.Figure()



fig = px.bar(title='COVID-19: India')



fig.add_trace(go.Bar(

    y=COVID_India_['State/UnionTerritory'],

    x=COVID_India_['Total Confirmed cases'],

    name='Confirmed',

    orientation='h',

    marker=dict(

        color='rgb(0, 60, 179)',

        line=dict(color='black')

    )

))



fig.add_trace(go.Bar(

    y=COVID_India_['State/UnionTerritory'],

    x=COVID_India_['Cured'],

    name='Cured',

    orientation='h',

    marker=dict(

        color='green',

        line=dict(color='black')

    )

))



fig.add_trace(go.Bar(

    y=COVID_India_['State/UnionTerritory'],

    x=COVID_India_['Death'],

    name='Death',

    orientation='h',

    marker=dict(

        color='rgb(179, 0, 0)',

        line=dict(color='black')

    )

))

fig.update_layout(barmode='stack')

fig.show()
COVID_India_
state_label=list(COVID_India_['State/UnionTerritory'].unique())

labels=[]

state=[]

values=[]

country=[]

for i in state_label:

    country.extend(('India','India','India'))

    labels.extend(('Confirmed','Cured','Death'))

    state.extend([i,i,i])

    values.extend([int(COVID_India_[COVID_India_['State/UnionTerritory']==i]['Total Confirmed cases']),

                 int(COVID_India_[COVID_India_['State/UnionTerritory']==i]['Cured']),

                 int(COVID_India_[COVID_India_['State/UnionTerritory']==i]['Death'])])
import plotly.graph_objects as go

df = pd.DataFrame(dict(C=country,L=labels, S=state, V=values))



fig = px.sunburst(df, path=['C','S','L'], values='V')



fig.update_layout(

    margin = dict(t=5, l=5, r=5, b=5)

)

fig.update_layout(title_text="COVID-19: India",

                  title_font_size=20)

fig.data[0].marker.line.width = 1

fig.data[0].marker.line.color = "white"

#fig.update_layout(uniformtext=dict(minsize=10))

fig.update_traces(

        go.Sunburst(hovertemplate='<b>%{label} </b><b>%{value:,.0f}</b>'),

        insidetextorientation='radial',       

    )

fig.show()
date_index2 = pd.date_range('1/30/2020', periods=len(COVID_India.groupby('Date').sum().index), freq='D')



fig = px.line(title='COVID-19: India')

fig.add_trace(go.Line(y= COVID_India.groupby('Date').sum()["Total Confirmed cases"],x=date_index2,name='Confirmed'))

fig.add_trace(go.Line(y= COVID_India.groupby('Date').sum()["Cured"],x=date_index2,name='Cured'))

fig.add_trace(go.Line(y= COVID_India.groupby('Date').sum()["Death"],x=date_index2,name='Death'))

fig.show()
# Latitude & longtitude detected using google.geocoder

# Already added columns into csv file

lat_long={'Andhra Pradesh':[15.9240905,80.1863809],

'Delhi'	:[28.6517178	,77.2219388],

'Haryana':	[29,	76],

'Karnataka'	:[14.5203896	,75.7223521],

'Kerala'	:[10.3528744,	76.5120396],

'Maharashtra':[	19.531932,	76.0554568],

'Odisha'	:[20.5431241	,84.6897321],

'Punjab':	[30.9293211,	75.5004841],

'Rajasthan'	:[26.8105777,	73.7684549],

'Tamil Nadu'	:[10.9094334	,78.3665347],

'Telengana':	[17.329125	,78.5822228],

'Union Territory of Jammu and Kashmir':	[33.91667,	76.66667],

'Union Territory of Ladakh':	[34.33333	,77.41667],

'Uttar Pradesh'	:[27.1303344	,80.859666],

'Uttarakhand'	:[30.09199355,	79.32176659]}



# Created Latitude & Longtitude columns for Folium Map

#COVID_India['Latitude']=COVID_India['State/UnionTerritory'].apply(lambda x : lat_long[x][0])

#COVID_India['Longtitude']=COVID_India['State/UnionTerritory'].apply(lambda x : lat_long[x][1])

#COVID_India
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(

    go.Scatter(x=COVID_India['Date'], y=COVID_India[COVID_India['State/UnionTerritory']=='Maharashtra']['Total Confirmed cases'],name='Confirmed: India',line=dict(color='blue')))
COVID_India_dates=list(COVID_India['Date'].unique())

COVID_India_state=list(COVID_India['State/UnionTerritory'].unique())

def rad_india(i):

    x = (i>150000 and 30) or(i>50000 and 20) or (i>10000 and 10) or (i>50000 and 5) or 1

    return(x)



# Folium plugins

from folium import IFrame

from folium.plugins import MarkerCluster

from folium.plugins import HeatMapWithTime

from folium.plugins import TimestampedGeoJson    

from IPython.display import display



# Define initial location

map_India = folium.Map(

    location=[28.7041, 77.1025],

    tiles='cartodbdark_matter',

    zoom_start=5

)



# Storing coordinates, popup, time in features list



features=[{'type': 'Feature',

         'geometry': {'type': 'Point','coordinates': [COVID_India['Longitude'][col], COVID_India['Latitude'][col]]},

         'properties': {'time': COVID_India['Date'][col],'icon': 'circle','iconstyle': {'fillColor': 'red','fillOpacity': 0.6,'stroke': 'false',

         'radius': rad_india(COVID_India['Total Confirmed cases'][col])},'style': {'weight': 0},'popup':'<b>State:{}<br>Confirmed:{}<br><b>Cured:{}<br>Death:{}'.format(COVID_India['State/UnionTerritory'][col],COVID_India['Total Confirmed cases'][col],COVID_India['Cured'][col],COVID_India['Death'][col])}}

          for col in range(COVID_India.shape[0])]



TimestampedGeoJson(

            {'type': 'FeatureCollection', 'features': features},

            period='P1D',

            auto_play=False,

            min_speed=3,

            max_speed=4,

            loop=False,

            loop_button=True,

            date_options='DD-MM-YYYY',

            ).add_to(map_India)



map_India                                                                                     
#TimestampedGeoJson Date input



COVID_dates= list(COVID_confirmed.columns[30:])



def rad_global(i):

    x = (i>1000000 and 30) or(i>500000 and 20) or (i>200000 and 15) or (i>100000 and 10)or (i>10000 and 5) or 2

    return(x)



# Folium plugins

from folium import IFrame

from folium.plugins import MarkerCluster

from folium.plugins import HeatMapWithTime

from folium.plugins import TimestampedGeoJson    

from IPython.display import display



# Define initial location

map_world = folium.Map(

    location=[28.7041, 77.1025],

    tiles='cartodbdark_matter',

    zoom_start=3,

)



# Storing coordinates, popup, time in features list



features=[{'type': 'Feature',

         'geometry': {'type': 'Point','coordinates':[COVID_confirmed['Long'][col],COVID_confirmed['Lat'][col]]},

         'properties': {'time': date,'icon': 'circle','iconstyle': {'fillColor': 'red','fillOpacity': 0.6,'stroke': 'false',

         'radius': rad_global(COVID_confirmed[date][col])},'style': {'weight': 0},'popup':'<b>Country:{}</b><br>Confirmed:{}<br>Death:{}'

         .format(COVID_confirmed['Country/Region'][col],COVID_confirmed[date][col],COVID_death[date][col])}}

         for date in COVID_dates for col in range(COVID_confirmed.shape[0]) if (COVID_confirmed[date][col])>0]

        

TimestampedGeoJson(

            {'type': 'FeatureCollection', 'features': features},

            period='P1D',

            auto_play=False,

            min_speed=1,

            max_speed=1,

            loop=False,

            loop_button=True,

            date_options='DD-MM-YYYY',

            ).add_to(map_world)



map_world          