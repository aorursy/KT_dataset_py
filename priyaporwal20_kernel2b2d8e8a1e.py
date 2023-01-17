#import packages

from __future__ import print_function

import pandas as pd

import numpy as np



from ipywidgets import interact, interactive, fixed, interact_manual

import ipywidgets as widgets
# loading data right from the source:

death_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv')

confirmed_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')

recovered_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv')

country_df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/web-data/data/cases_country.csv')
death_df.head()
confirmed_df.head()
recovered_df.head()
country_df.head()
# data cleaning - renaming

country_df.columns = map(str.lower, country_df.columns)

recovered_df.columns = map(str.lower, recovered_df.columns)

death_df.columns = map(str.lower, death_df.columns)

confirmed_df.columns = map(str.lower, confirmed_df.columns)
confirmed_df = confirmed_df.rename(columns={'province/state': 'state', 'country/region': 'country'})

recovered_df = confirmed_df.rename(columns={'province/state': 'state', 'country/region': 'country'})

death_df = death_df.rename(columns={'province/state': 'state', 'country/region': 'country'})

country_df = country_df.rename(columns={'country_region': 'country'})
sorted_country_df = country_df.sort_values('confirmed', ascending=False).head(5)
sorted_country_df
def highlight_col(x):

    r = 'background-color: red'

    p = 'background-color: purple'

    g = 'background-color: grey'

    temp_df = pd.DataFrame('', index=x.index, columns = x.columns)

    temp_df.iloc[:,4] = p

    temp_df.iloc[:,5] = r

    temp_df.iloc[:,6] = g

    return temp_df



sorted_country_df.style.apply(highlight_col, axis=None)
! pip install plotly
import plotly.express as px
fig = px.scatter(sorted_country_df.head(10), x='country', y='confirmed', size='confirmed',

                color='country', hover_name='country', size_max=60)

#fig.update_layout()

fig.show()
import plotly.graph_objects as go



def plot_cases_for_country(country):

    labels = ['confirmed', 'deaths']

    colors = ['blue', 'red']

    mode_size = [6,8]

    line_size = [4,5]



    df_list = [confirmed_df, death_df]



    fig = go.Figure()



    for i, df in enumerate(df_list):

        if country == 'World' or country == 'world':

            x_data = np.array(list(df.iloc[:, 5:].columns))

            y_data = np.sum(np.asarray(df.iloc[:, 5:]), axis=0)

        

        else:

            x_data = np.array(list(df.iloc[:, 5:].columns))

            y_data = np.sum(np.asarray(df[df['country']==country].iloc[:, 5:]), axis=0)

        

        fig.add_trace(go.Scatter(x=x_data, y=y_data, mode='lines+markers',

                                name=labels[i],

                                line = dict(color=colors[i], width=line_size[i]),

                                connectgaps = True,

                                text = "Total " + str(labels[i]) + ": " + str(y_data[-1])

                                ))

    fig.show()

        

#plot_cases_for_country('China')



interact(plot_cases_for_country, country='World')
!pip install folium

import folium

world_map = folium.Map(location=[11,0], tiles='cartodbpositron', zoom_start=2, max_zoom=6, min_zoom=2)



for i in range(len(confirmed_df)):

    folium.Circle(

    location=[confirmed_df.iloc[i]['lat'],confirmed_df.iloc[i]['long']],

    fill = True,

    radius = (int((np.log(confirmed_df.iloc[i,-1]+1.00001)))+0.2)*50000,

    fill_color = 'blue',

    color = 'red',

    tooltip = "<div style='margin: 0; background-color: black; color: white;'>"+

                    "<h4 style='text-align:center;font-weight: bold'>"+confirmed_df.iloc[i]['country'] + "</h4>"

                    "<hr style='margin:10px;color: white;'>"+

                    "<ul style='color: white;;list-style-type:circle;align-item:left;padding-left:20px;padding-right:20px'>"+

                        "<li>Confirmed: "+str(confirmed_df.iloc[i,-1])+"</li>"+

                        "<li>Deaths:   "+str(death_df.iloc[i,-1])+"</li>"+

                        "<li>Death Rate: "+ str(np.round(death_df.iloc[i,-1]/(confirmed_df.iloc[i,-1]+1.00001)*100,2))+ "</li>"+

                    "</ul></div>"

    ).add_to(world_map)



world_map