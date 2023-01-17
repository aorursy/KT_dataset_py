import matplotlib.pyplot as plt

import matplotlib.image as mpimg

import numpy as np

import pandas as pd

import folium

import seaborn as sns

import warnings

import plotly.express as px

import plotly.graph_objects as go

from IPython.display import HTML

from fbprophet import Prophet

from folium.plugins import HeatMapWithTime

warnings.filterwarnings('ignore')

%matplotlib inline
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))




COVID19_line_list_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_line_list_data.csv")

COVID19_open_line_list = pd.read_csv("../input/novel-corona-virus-2019-dataset/COVID19_open_line_list.csv")

covid_data = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",parse_dates=True)

time_series_covid_19_confirmed = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_confirmed.csv")

time_series_covid_19_deaths = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_deaths.csv")

time_series_covid_19_recovered = pd.read_csv("../input/novel-corona-virus-2019-dataset/time_series_covid_19_recovered.csv")

img=mpimg.imread('../input/corona-local/China poll graphic.png')

geo_l = pd.read_csv("../input/world-coordinates/world_coordinates.csv")

# geo_l[geo_l['Country']=='China']

geo_l = geo_l.append({'Code' : 'CN' 

                      , 'Country' : 'Mainland China'

                      ,'latitude' : '35.86166'

                      ,'longitude' : '104.195397'

                     } , ignore_index=True)
covid_data['ObservationDate'].max()
country_CDR=pd.DataFrame(covid_data.groupby(['ObservationDate','Country/Region'],as_index=False)[['Confirmed','Deaths','Recovered']].sum())

geo_w_d_CDR=country_CDR.merge(geo_l,left_on='Country/Region',right_on='Country')

#geo_w_d_CDR=geo_w_d_CDR.drop(['country','name'],axis=1)

#geo_w_d_CDR

# create map and display it

geo_w_m = folium.Map(location=[10, -20] ,zoom_start=2.3,tiles='OpenStreetMap')



for lat, lon, value, name in zip(geo_w_d_CDR['latitude'], geo_w_d_CDR['longitude'], geo_w_d_CDR['Confirmed'], geo_w_d_CDR['Country/Region']):

    folium.CircleMarker([lat, lon],

                        radius=value/5000,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Confirmed Cases</strong>: ' + str(value) + '<br>'),

                        color='gray',

                        

                        fill_color='gray',

                        fill_opacity=0.1 ).add_to(geo_w_m)

geo_w_m
# create map and display it



deaths = geo_w_d_CDR.loc[geo_w_d_CDR['Deaths'] > 0]

geo_w_m_D = folium.Map(location=[0, -30] ,zoom_start=2.3,tiles='OpenStreetMap')



for lat, lon, value, name in zip(deaths['latitude'], deaths['longitude'], deaths['Deaths'], deaths['Country/Region']):

    folium.CircleMarker([lat, lon],

                        radius=value/150,

                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'

                                '<strong>Reported Deaths</strong>: ' + str(value) + '<br>'),

                        color='#CD5C5C',

                        

                        fill_color='#CD5C5C',

                        fill_opacity=0.1 ).add_to(geo_w_m_D)

geo_w_m_D
recovered = geo_w_d_CDR.loc[geo_w_d_CDR['Recovered'] > 0]

geo_w_m_R = folium.Map(location=[10, -20] ,zoom_start=2.3,tiles='OpenStreetMap')



for lat, lon, count, country in zip(recovered['latitude'], recovered['longitude'], recovered['Recovered'], recovered['Country/Region']):

    folium.CircleMarker([lat, lon],

                        radius=8,

                        popup = ('<strong>Country</strong>: ' + str(country).capitalize() + '<br>'

                                '<strong>Recovered</strong>: ' + str(count) + '<br>'),

                        color='Green',

                        

                        fill_color='Green',

                        fill_opacity=0.1 ).add_to(geo_w_m_R)

geo_w_m_R
confirmed_latl = time_series_covid_19_confirmed[["Province/State","Lat","Long","Country/Region"]]

df_temp = covid_data.copy()

df_temp['Country/Region'].replace({'Mainland China': 'China'}, inplace=True)

df_latlong = pd.merge(df_temp, confirmed_latl, on=["Country/Region", "Province/State"])
# # import ipywidgets

# # base_map = folium.Map(location=[10, -20] ,zoom_start=2.3,tiles='CartoDB Dark_Matter')

# # HeatMapWithTime(df_latlong[['Lat','Long']]

# #                 , radius=5

# #                 , index=df_latlong['ObservationDate']

# #                 , auto_play=True

# #                 , gradient={0.2: 'blue', 0.4: 'lime', 0.6: 'orange', 1: 'red'}

# #                 , min_opacity=0.5, max_opacity=0.8, use_local_extrema=True).add_to(base_map)

# # base_map



fig = px.density_mapbox(df_latlong,

                        

                        lat="Lat", 

                        lon="Long", 

                        hover_name="Country/Region", 

                        hover_data=["Confirmed","Deaths","Recovered"], 

                        animation_frame="ObservationDate",

                        color_continuous_scale="thermal",

                        radius=7, 

                        zoom=0,height=700)

fig.update_layout(title='Worldwide Corona Virus Cases Time Lapse - Confirmed, Deaths, Recovered',

                  font=dict(family="Courier New, monospace",

                            size=18,

                            color="#7f7f7f")

                 )

fig.update_layout(

    margin ={'l':0,'t':0,'b':0,'r':0},

    mapbox = {

#         'center': {'lon': -10, 'lat': 10},

        'style': "carto-darkmatter",

#         'center': {'lon': -20, 'lat': -20},

        'zoom':1

        

        

       

        }

)

fig.update_layout(autosize=False,width=600,

    height=800,mapbox_center_lon=82)





fig.show()
covid_data_n=covid_data.drop(columns=['Last Update','Province/State'],axis=1)

covid_data_n.groupby(['Country/Region','ObservationDate']).sum()

country_t=pd.DataFrame(covid_data_n.groupby(['Country/Region'])['Confirmed'].sum()).sort_values(by=['Confirmed'],ascending=False)[:11]

# country_t.index
covid_lines=covid_data[covid_data['Country/Region'].isin(country_t.index)]

covid_lines=covid_lines.drop(columns=['SNo'],axis=True)

t=pd.DataFrame(covid_lines.groupby(['Country/Region','ObservationDate'],as_index=False).sum())

Zeros = t[t.Confirmed == 0]

t=t.drop(Zeros.index)

t.sort_values("ObservationDate", axis = 0, ascending = True, 

                 inplace = True,) 



# covid_lines['ObservationDate'] = covid_lines['ObservationDate'].dt.date
t_all=country_t=pd.DataFrame(t.groupby(['ObservationDate']).sum())

# t_all=t_all.drop(columns=['Country/Region'],axis=1)

# t_all=t_all.set_index('ObservationDate')





f, ax = plt.subplots(1,1, figsize=(14,8))

k = sns.lineplot(data=t_all)

plt.xticks(rotation=90)

plt.title('Global Situation')

plt.xlabel('Date')

plt.ylabel('Cases')

plt.show()
f, ax = plt.subplots(1,1, figsize=(14,8))

g = sns.lineplot(x='ObservationDate', y='Confirmed', hue='Country/Region', data=t)

plt.xticks(rotation=90)

plt.title('Top 10 Countries - Cases Confirmed By Date')

plt.show()  
f, ax = plt.subplots(1,1, figsize=(14,8))

g = sns.lineplot(x='ObservationDate', y='Deaths', hue='Country/Region', data=t)

plt.xticks(rotation=90)

plt.title('Top 10 Countries - Deaths Confirmed By Date')



plt.show() 
f, ax = plt.subplots(1,1, figsize=(14,8))

g = sns.lineplot(x='ObservationDate', y='Recovered', hue='Country/Region', data=t)

plt.xticks(rotation=90)

plt.title('Top 10 Countries - Recoveries Confirmed By Date')



plt.show() 
f, ax = plt.subplots(1,1, figsize=(14,8))

plt.imshow(img)
covid_forecast=pd.DataFrame(covid_data.groupby(['ObservationDate'],as_index=False)['Confirmed'].sum())

covid_forecast.columns=['ds','y']



model = Prophet()

model.fit(covid_forecast)



future = model.make_future_dataframe(periods=14)

# future.tail()

forecast = model.predict(future)

# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()

# fig1 = model.plot(forecast)
from fbprophet.plot import plot_plotly

import plotly.offline as py

py.init_notebook_mode()

fig = plot_plotly(model, forecast) 

py.iplot(fig)
# HTML('''<script>

# code_show=true; 

# function code_toggle() {

#  if (code_show){

#  $('div.input').hide();

#  } else {

#  $('div.input').show();

#  }

#  code_show = !code_show

# } 

# $( document ).ready(code_toggle);

# </script>

# <form action="javascript:code_toggle()"><input type="submit" value="Hide/See raw code."></form>''')