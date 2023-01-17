# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import requests

import json

import numpy as np

import pandas as pd



mundo = requests.get('https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json')

geo_mundo = json.loads(mundo.content)

data_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

deaths_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')
data_confirmed = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')



ultima_fecha_cl = data_confirmed.columns

ultima_fecha_cl= ultima_fecha_cl[-1]





data_confirmed.loc[data_confirmed['Country/Region'] == 'US', "Country/Region"] = 'United States of America'



data_confirmed.loc[data_confirmed['Country/Region'] == 'US', "Country/Region"] = 'United States of America'

data_confirmed.loc[data_confirmed['Country/Region'] == 'Congo (Kinshasa)', "Country/Region"] = 'Democratic Republic of the Congo'



#AGREGAR French Guiana COMO PAIS Y NO PROVICNCIA DE FRANCIA PERO SE SUMAR A EL TOTAL DE FRANCIA IGUAL

data_confirmed = data_confirmed.append({'Country/Region':'French Guiana',ultima_fecha_cl: int(data_confirmed[data_confirmed['Province/State']=='French Guiana'][ultima_fecha_cl])}, ignore_index=True)



data_confirmed.loc[data_confirmed['Country/Region'] == "Cote d'Ivoire", "Country/Region"] = 'Ivory Coast'

data_confirmed.loc[data_confirmed['Country/Region'] == 'Congo (Brazzaville)', "Country/Region"] = 'Republic of the Congo'

data_confirmed.loc[data_confirmed['Country/Region'] == 'Tanzania', "Country/Region"] = 'United Republic of Tanzania'

data_confirmed.loc[data_confirmed['Country/Region'] == 'Korea, South', "Country/Region"] = 'South Korea'



d = data_confirmed.groupby(['Country/Region']).sum()



paises = data_confirmed['Country/Region'].drop_duplicates()

paises = sorted(paises)



data_mundo_mapa = pd.DataFrame({'Country': paises,'Casos':d[ultima_fecha_cl]})

data_mundo_mapa.head()
recoveries_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')



ultima_fecha_cl = deaths_data.columns

ultima_fecha_cl= ultima_fecha_cl[-1]





recoveries_df.loc[recoveries_df['Country/Region'] == 'US', "Country/Region"] = 'United States of America'



recoveries_df.loc[recoveries_df['Country/Region'] == 'US', "Country/Region"] = 'United States of America'

recoveries_df.loc[recoveries_df['Country/Region'] == 'Congo (Kinshasa)', "Country/Region"] = 'Democratic Republic of the Congo'



#AGREGAR French Guiana COMO PAIS Y NO PROVICNCIA DE FRANCIA PERO SE SUMAR A EL TOTAL DE FRANCIA IGUAL

recoveries_df = recoveries_df.append({'Country/Region':'French Guiana',ultima_fecha_cl: int(data_confirmed[data_confirmed['Province/State']=='French Guiana'][ultima_fecha_cl])}, ignore_index=True)



recoveries_df.loc[recoveries_df['Country/Region'] == "Cote d'Ivoire", "Country/Region"] = 'Ivory Coast'

recoveries_df.loc[recoveries_df['Country/Region'] == 'Congo (Brazzaville)', "Country/Region"] = 'Republic of the Congo'

recoveries_df.loc[recoveries_df['Country/Region'] == 'Tanzania', "Country/Region"] = 'United Republic of Tanzania'

recoveries_df.loc[recoveries_df['Country/Region'] == 'Korea, South', "Country/Region"] = 'South Korea'







d2 = recoveries_df.groupby(['Country/Region']).sum()



paises = recoveries_df['Country/Region'].drop_duplicates()

paises = sorted(paises)

v = d2[ultima_fecha_cl].apply(str)



data_mundo_mapa_rec = pd.DataFrame({'Country': paises,'Recuperados':v})

data_mundo_mapa_rec.head()
deaths_data = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')





ultima_fecha_cl = deaths_data.columns

ultima_fecha_cl= ultima_fecha_cl[-1]





deaths_data.loc[deaths_data['Country/Region'] == 'US', "Country/Region"] = 'United States of America'



deaths_data.loc[deaths_data['Country/Region'] == 'US', "Country/Region"] = 'United States of America'

deaths_data.loc[deaths_data['Country/Region'] == 'Congo (Kinshasa)', "Country/Region"] = 'Democratic Republic of the Congo'



#AGREGAR French Guiana COMO PAIS Y NO PROVICNCIA DE FRANCIA PERO SE SUMAR A EL TOTAL DE FRANCIA IGUAL

deaths_data = deaths_data.append({'Country/Region':'French Guiana',ultima_fecha_cl: int(data_confirmed[data_confirmed['Province/State']=='French Guiana'][ultima_fecha_cl])}, ignore_index=True)



deaths_data.loc[deaths_data['Country/Region'] == "Cote d'Ivoire", "Country/Region"] = 'Ivory Coast'

deaths_data.loc[deaths_data['Country/Region'] == 'Congo (Brazzaville)', "Country/Region"] = 'Republic of the Congo'

deaths_data.loc[deaths_data['Country/Region'] == 'Tanzania', "Country/Region"] = 'United Republic of Tanzania'

deaths_data.loc[deaths_data['Country/Region'] == 'Korea, South', "Country/Region"] = 'South Korea'







d2 = deaths_data.groupby(['Country/Region']).sum()



paises = deaths_data['Country/Region'].drop_duplicates()

paises = sorted(paises)



data_mundo_mapa_death = pd.DataFrame({'Country': paises,'Fallecidos':d2[ultima_fecha_cl]})

data_mundo_mapa_death.head()
data_cd = pd.merge(data_mundo_mapa, data_mundo_mapa_death, on='Country')

data_cdr =  pd.merge(data_cd, data_mundo_mapa_rec, on='Country')

data_cdr.head()
import plotly.graph_objects as go

fig = go.Figure(go.Choroplethmapbox(geojson=geo_mundo, locations=data_cdr.Country, z=data_cdr.Casos,

                                    colorscale="Viridis", zmin=0, zmax=100000,

                                    featureidkey="properties.name",

                                    colorbar = dict(thickness=20, ticklen=3),

                                    marker_opacity=0.2, marker_line_width=0,  

                                    text=data_cdr['Fallecidos'],

                                      hovertemplate = '<b>Country</b>: <b>'+data_cdr['Country']+'</b>'+

                                            '<br><b>Cases </b>: %{z}<br>'+

                                            '<b>Deceased: </b>:%{text}<br>'+

                                            '<b>Recovered</b>: <b>'+data_cdr['Recuperados']  ))



fig.update_layout(mapbox_style="carto-positron",

                  mapbox_zoom=1,height=700,mapbox_center = {"lat": 0, "lon": 0})

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

fig.show()