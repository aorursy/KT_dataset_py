import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

from plotly.offline import iplot, init_notebook_mode

import plotly.express as px

import plotly.graph_objs as go

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)



import folium

plt.style.use('ggplot')





import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df_daily = pd.read_csv('../input/dailydata/03-23-2020.csv')

df_time_series_confirmed = pd.read_csv('../input/timeseries/time_series_covid19_confirmed_global.csv')

df_time_series_death = pd.read_csv('../input/timeseries/time_series_covid19_deaths_global.csv')

df_dolar_brazil = pd.read_csv('../input/financialdata/USD_BRL.csv')
df_daily.head(5)
df_time_series_death.head()
df_dolar_brazil.head()
top_country_death = df_daily[['Country_Region','Deaths']]

top_country_death = top_country_death.groupby(['Country_Region'], as_index=False).sum()

top_country_death = top_country_death.sort_values(by=['Deaths'], ascending=False).head(10)

fig = px.bar(top_country_death, y='Deaths', x='Country_Region', color='Country_Region', orientation="v")



fig.update_layout(

    title={

        'text': 'Top 10 países com o maior número de óbitos',

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.update_xaxes(title_text='País')

fig.update_yaxes(title_text='Número de óbitos')



fig.show()
top_country_death = df_daily[['Country_Region','Deaths']]

top_country_death = top_country_death.groupby(['Country_Region'], as_index=False).sum()

top_country_death = top_country_death.sort_values(by=['Deaths'], ascending=False).head(10)



coronavirus_worldwide = df_time_series_death.drop(columns = ['Province/State','Lat', 'Long'])

coronavirus_worldwide = coronavirus_worldwide.groupby(['Country/Region']).sum()

coronavirus_worldwide = coronavirus_worldwide.sort_values(by = coronavirus_worldwide.columns[-1], ascending = False)

coronavirus_worldwide = coronavirus_worldwide.head(10)

coronavirus_worldwide = coronavirus_worldwide.transpose()

coronavirus_worldwide.iplot(asFigure=True,kind='line',xTitle='Data',yTitle='Número de Óbitos',title='Série temporal dos top 10 países com o maior número de óbitos')
df_groupby = df_daily.groupby(['Country_Region'], as_index=False).sum()



df_without_death = df_groupby.copy()

df_without_death = df_without_death.loc[(df_without_death['Deaths'] == 0) & (df_without_death['Confirmed'] == 0)]

df_without_death_and_confirmed = df_groupby.copy()

df_without_death_and_confirmed = df_without_death_and_confirmed.loc[(df_without_death_and_confirmed['Deaths'] == 0) & (df_without_death_and_confirmed['Confirmed'] >= 0)]

df_death_and_confirmed = df_groupby.copy()

df_death_and_confirmed = df_death_and_confirmed.loc[(df_death_and_confirmed['Deaths'] >= 1)]
fig = go.Figure(data=go.Scattergeo(

        lon = df_without_death_and_confirmed['Long_'],

        lat = df_without_death_and_confirmed['Lat'],

        mode = 'markers',

        hovertext=df_without_death_and_confirmed['Country_Region'],

        marker = dict(

            color = 'orange',

            line = {'width': 1,'color': 'orange'},

            sizemode = 'area')

        ))



fig.update_layout(

        title={

        'text': 'Países em que não há ocorrência de óbito por coronavirus <br> (entre os que já registraram casos confirmados)',

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
fig = go.Figure(data=go.Scattergeo(

        lon = df_death_and_confirmed['Long_'],

        lat = df_death_and_confirmed['Lat'],

        mode = 'markers',

        hovertext=df_death_and_confirmed['Country_Region']+ ' - Óbitos: ' ++ df_death_and_confirmed['Deaths'].astype(str),

        marker = dict(

            color = 'red',

            line = {'width': 1,'color': 'red'},

            sizemode = 'area')

        ))



fig.update_layout(

        title={

        'text': 'Países em que há ocorrência de óbito por coronavirus',

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})

fig.show()
df_groupby_south_america = df_groupby.copy()



df_groupby_south_america = df_groupby_south_america.loc[(df_groupby_south_america['Country_Region'] == "Brazil") | (df_groupby_south_america['Country_Region'] == "Argentina") 

                          | (df_groupby_south_america['Country_Region'] == "Bolivia") | (df_groupby_south_america['Country_Region'] == "Uruguay") 

                          | (df_groupby_south_america['Country_Region'] == "Chile") | (df_groupby_south_america['Country_Region'] == "Paraguay")

                          | (df_groupby_south_america['Country_Region'] == "Peru")| (df_groupby_south_america['Country_Region'] == "Colombia")

                          | (df_groupby_south_america['Country_Region'] == "Venezuela")| (df_groupby_south_america['Country_Region'] == "Ecuador")

                          | (df_groupby_south_america['Country_Region'] == "Guyana") | (df_groupby_south_america['Country_Region'] == "Suriname")

                          | (df_groupby_south_america['Country_Region'] == "French Guiana")]
top_country_south_america_death = df_groupby_south_america.loc[(df_groupby_south_america['Deaths'] >= 1)]

top_country_south_america_death = top_country_south_america_death.sort_values(by=['Deaths'], ascending=False).head(15)

fig = px.bar(top_country_south_america_death, y='Deaths', x='Country_Region', color='Country_Region', orientation="v")



fig.update_layout(

    title={

        'text': 'Países sul-americanos com o maior número de óbitos por coronavirus',

        'y':0.95,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.update_xaxes(title_text='País')

fig.update_yaxes(title_text='Número de óbitos')



fig.show()
df_groupby_south_america.loc[df_groupby_south_america['Deaths'] >=1, 'Having Death'] = 'yes'

df_groupby_south_america.loc[(df_groupby_south_america['Deaths'] == 0) & (df_groupby_south_america['Confirmed'] == 0), 'Having Death'] = 'still safe'

df_groupby_south_america.loc[(df_groupby_south_america['Deaths'] == 0) & (df_groupby_south_america['Confirmed'] >= 1), 'Having Death'] = 'only confirmed'
colors = {

 'yes': 'red',

 'still safe': 'green',

 'only confirmed': 'orange'

}



south_america_map = folium.Map(

    location=[-14.235,-51.9253],

    zoom_start=3

)



for _, sa in df_groupby_south_america.iterrows():

    if sa['Having Death'] in colors.keys():

        folium.Marker(

            location=[sa['Lat'], sa['Long_']],

            tooltip=str(sa['Confirmed'])+ str(" Confirmed")+" | "+str(sa['Recovered'])+ str(" Recovered")+" | "+str(sa['Deaths'])+ str(" Deaths"),

            icon=folium.Icon(color=colors[sa['Having Death']])

        ).add_to(south_america_map)



south_america_map
df_brazil_china_italy = df_time_series_confirmed.copy()

df_brazil_china_italy = df_brazil_china_italy.drop(columns = ['Province/State','Lat', 'Long'])

df_brazil_china_italy = df_brazil_china_italy.groupby(['Country/Region'], as_index=False).sum()

df_brazil_china_italy = df_brazil_china_italy.loc[(df_brazil_china_italy['Country/Region'] == "Brazil") 

                                                  | (df_brazil_china_italy['Country/Region'] == "China") 

                                                  | (df_brazil_china_italy['Country/Region'] == "Italy")]

df_brazil_china_italy = df_brazil_china_italy.groupby(['Country/Region']).sum()

df_brazil_china_italy = df_brazil_china_italy.sort_values(by = df_brazil_china_italy.columns[-1], ascending = False)

df_brazil_china_italy = df_brazil_china_italy.head(10)

df_brazil_china_italy = df_brazil_china_italy.transpose()

df_brazil_china_italy.iplot(asFigure=True,kind='line',xTitle='Data',yTitle='Número de Casos Confirmados',title='Crescimento do número de casos confirmados entre Brasil, China e Itália')
df_brazil_china_italy = df_time_series_death.copy()

df_brazil_china_italy = df_brazil_china_italy.drop(columns = ['Province/State','Lat', 'Long'])

df_brazil_china_italy = df_brazil_china_italy.groupby(['Country/Region'], as_index=False).sum()

df_brazil_china_italy = df_brazil_china_italy.loc[(df_brazil_china_italy['Country/Region'] == "Brazil") 

                                                  | (df_brazil_china_italy['Country/Region'] == "China") 

                                                  | (df_brazil_china_italy['Country/Region'] == "Italy")]

df_brazil_china_italy = df_brazil_china_italy.groupby(['Country/Region']).sum()

df_brazil_china_italy = df_brazil_china_italy.sort_values(by = df_brazil_china_italy.columns[-1], ascending = False)

df_brazil_china_italy = df_brazil_china_italy.head(10)

df_brazil_china_italy = df_brazil_china_italy.transpose()

df_brazil_china_italy.iplot(asFigure=True,kind='line',xTitle='Data',yTitle='Número de Óbitos',title='Crescimento do número de óbitos entre Brasil, China e Itália')
df_time_series_confirmed_brazil = df_time_series_confirmed.copy()

df_time_series_confirmed_brazil = df_time_series_confirmed_brazil.drop(columns = ['Province/State','Lat', 'Long'])

df_time_series_confirmed_brazil = df_time_series_confirmed_brazil.groupby(['Country/Region'], as_index=False).sum()

df_time_series_confirmed_brazil = df_time_series_confirmed_brazil.loc[(df_time_series_confirmed_brazil['Country/Region'] == "Brazil")]

df_time_series_confirmed_brazil = df_time_series_confirmed_brazil.groupby(['Country/Region']).sum()

df_time_series_confirmed_brazil = df_time_series_confirmed_brazil.sort_values(by = df_time_series_confirmed_brazil.columns[-1], ascending = False)

df_time_series_confirmed_brazil = df_time_series_confirmed_brazil.transpose()

df_time_series_confirmed_brazil.iplot(asFigure=True,kind='line',xTitle='Data',yTitle='Número de casos confirmados',title='Número de pessoas confirmadas com coronavirus no Brasil')
df_dolar_brazil = df_dolar_brazil.stack().str.replace(',','.').unstack()

df_dolar_brl = df_dolar_brazil.copy()

df_dolar_brl = df_dolar_brl.drop(columns = ['Abertura','Máxima','Mínima','Var%'])

df_dolar_brl['Último'] = df_dolar_brl['Último'].astype(float)

df_dolar_brl['Data'] = pd.to_datetime(df_dolar_brl['Data'], format='%d.%m.%Y')

df_dolar_brl = df_dolar_brl.sort_values(by='Data',ascending=True)

df_dolar_brl['Data1'] = df_dolar_brl['Data'].dt.strftime('%m/%d/%Y')
fig = go.Figure([go.Scatter(x=df_dolar_brl['Data1'], y=df_dolar_brl['Último'])])

fig.update_layout(

        title={

        'text': 'Oscilação do dólar no Brasil',

        'y':0.9,

        'x':0.5,

        'xanchor': 'center',

        'yanchor': 'top'})



fig.update_xaxes(title_text='Data')

fig.update_yaxes(title_text='Valor do Dólar')



fig.show()