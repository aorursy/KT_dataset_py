# Imports

from __future__ import print_function



import pandas as pd

import numpy as np

import plotly.express as px

import plotly.graph_objects as go

from IPython.core.display import display, HTML



from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets
# Data used from John Hopkins COVID-19

df_deaths = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

df_cases = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

df_recovered = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

df_general = pd.read_csv('https://raw.githubusercontent.com/imdevskp/covid_19_jhu_data_web_scrap_and_cleaning/master/country_wise_latest.csv')
# Data cleaning

df_general.columns = map(str.lower, df_general.columns)

df_deaths.columns = map(str.lower, df_deaths.columns)

df_cases.columns = map(str.lower, df_cases.columns)

df_recovered.columns = map(str.lower, df_recovered.columns)
df_cases = df_cases.rename(columns= {'province/state': 'state', 'country/region': 'country'})

df_deaths = df_deaths.rename(columns= {'province/state': 'state', 'country/region': 'country'})

df_recovered = df_recovered.rename(columns= {'province/state': 'state', 'country/region': 'country'})

df_general = df_general.rename(columns= {'country/region': 'country'})
df_general.head()
df_deaths.head()
df_cases.head()
df_recovered.head()
df_general_sorted = df_general.sort_values('confirmed', ascending=False)
confirmed_total = int(df_general['confirmed'].sum())

deaths_total = int(df_general['deaths'].sum())

recovered_total = int(df_general['recovered'].sum())

active_total = int(df_general['active'].sum())
display(HTML("<div style = 'text-align:center; background-color: #f5faff; padding: 30px '>" +

             "<div style='color: cornflowerblue; font-size:30px;'><br><strong> Confirmed: "  + str(confirmed_total) +"</br></strong></div>" +

             "<div style='color: red; font-size:30px;margin-left:20px;'><br><strong> Deaths: " + str(deaths_total) + "</br></strong></div>"+

             "<div style='color: green; font-size:30px; margin-left:20px;'><br><strong> Recovered: " + str(recovered_total) + "</br></strong></div>"+

             "</div>")

       )

def country_cases(country):

    labels = ['confirmed', 'deaths']

    colors = ['cornflowerblue', 'red']

    mode_size = [6,8]

    line_size = [4,5]



    df_list = [df_cases, df_deaths]



    fig = go.Figure()



    for i, df in enumerate(df_list):

        if country == 'World' or country == 'world':

            x_data = np.array(list(df.iloc[:,5:].columns))

            y_data = np.sum(np.asarray(df.iloc[:,5:]), axis=0)

        else:

            x_data = np.array(list(df.iloc[:,5:].columns))

            y_data = np.sum(np.asarray(df[df['country']==country].iloc[:,5:]), axis=0)

        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers', name=labels[i],

                                line = dict(color=colors[i], width=line_size[i]),

                                connectgaps=True,

                                text='Total ' + str(labels[i]) + ':' + str(y_data[-1])

                                ))

    fig.show()
interact(country_cases, country='World');
def highlight_column(x):

    red = 'background-color: red'

    green = 'background-color: lime'

    blue = 'background-color: cornflowerblue'

    df_temp = pd.DataFrame('', index=x.index, columns=x.columns)

    df_temp.iloc[:,1] = blue

    df_temp.iloc[:,2] = red

    df_temp.iloc[:,3] = green

    return df_temp
df_general_sorted.head(10).style.apply(highlight_column, axis=None)
fig = px.scatter(df_general_sorted.head(15), x='country', y='confirmed', size='confirmed', color='country', hover_name='country', size_max=60)

fig.update_layout()

fig.show()
import folium
world_map = folium.Map(location=[11,0], tiles='cartodbpositron', zoom_start=2, max_zoom=6, min_zoom=2)



for i in range(len(df_cases)):

    folium.Circle(location=[df_cases.iloc[i]['lat'], df_cases.iloc[i]['long']],

    fill=True,

    radius=(int((np.log(df_cases.iloc[i,-1]+1.00001)))+0.2)*50000,

    fill_color='crimson',

    color='crimson',

    tooltip = "<div style='margin:0; background-color: firebrick; color: white;'>" +

                  "<h4 style='text-align:center; font-weight: bold'>" + df_cases.iloc[i]['country'] + "</h4>"

                  "<hr style='margin:10px; color: white;'>" +

                  "<ul style='color: white;;list-style-type: circle; align-item: left; padding-left: 20px; padding-right:20px'>" +

                      "<li>Total cases: " + str(df_cases.iloc[i,-1]) + "</li>" +

                      "<li>Deaths: " + str(df_deaths.iloc[i,-1]) + "</li>" +

                      "<li>Death rate : " + str(np.round(df_deaths.iloc[i,-1]/(df_cases.iloc[i,-1]+1.00001)*100,2)) + "% </li>" +

                      "</ul></div>",

    ).add_to(world_map)

    

world_map
df_general_sorted = df_general.sort_values('1 week % increase', ascending=False)



fig = px.bar(df_general_sorted, x=df_general_sorted['country'][:15], y=df_general_sorted['1 week % increase'][:15], title='Higher 1 week % increase', color=df_general_sorted['country'][:15],

             labels={'x': 'Country', 'y': '% increase', 'color': 'country'})

fig.show()