! pip install calmap

# essential libraries

import random

from datetime import timedelta  



# storing and anaysis

import numpy as np

import pandas as pd



# visualization

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import calmap

import folium



# color pallette

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



# html embedding

from IPython.display import Javascript

from IPython.core.display import display

from IPython.core.display import HTML
# importing datasets

covidData = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv', 

                         parse_dates=['Date'])

# replacing Mainland china with just China

covidData['Country/Region'] = covidData['Country/Region'].replace('Mainland China', 'China')

covidData.sample(6)
allCountries = covidData['Country/Region'].unique()
countries = ['Germany','US','Italy','Spain','United Kingdom','France','Switzerland']
display(HTML('<h1>Worldwide</h1>'))

# ------------------------------------------------------------------------------------------

temp = covidData.groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

temp['No. of Deaths to 100 Confirmed Cases'] = round(temp['Deaths']/temp['Confirmed'], 3)*100

temp['No. of Recovered to 100 Confirmed Cases'] = round(temp['Recovered']/temp['Confirmed'], 3)*100



fig_1 = px.area(temp, x="Date", y="Confirmed", color_discrete_sequence = [act])

fig_2 = px.area(temp, x="Date", y="Deaths", color_discrete_sequence = [dth])

fig_3 = px.line(temp, x="Date", y="No. of Deaths to 100 Confirmed Cases",  color_discrete_sequence=['#333333'])

fig_4 = px.line(temp, x="Date", y="No. of Recovered to 100 Confirmed Cases",  color_discrete_sequence=['#0f4c75'])



# ------------------------------------------------



# -------------------------------------------------------------------------------------------



temp = covidData.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan



temp = temp.groupby('Date')['Confirmed'].sum().reset_index()



fig_5 = px.bar(temp, x="Date", y="Confirmed", color_discrete_sequence = [act])

fig_5.update_layout(xaxis_rangeslider_visible=True)



# -------------------------------------------------------------------------------------------



temp = covidData.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan



temp = temp.groupby('Date')['Deaths'].sum().reset_index()



fig_6 = px.bar(temp, x="Date", y="Deaths", color_discrete_sequence = [dth])

fig_6.update_layout(xaxis_rangeslider_visible=True)







# ==========================================================================================



fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.08, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Deaths reported',

                                    'No. of new cases everyday','No. of new deaths everyday'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_5['data'][0], row=2, col=1)

fig.add_trace(fig_6['data'][0], row=2, col=2)



fig.update_layout(height=1200)

fig.show()



for country in countries:

    display(HTML(f'<h1>{country}</h1>'))

    # ------------------------------------------------------------------------------------------

    temp = covidData.loc[covidData['Country/Region']==country,:].groupby('Date')['Confirmed', 'Deaths', 'Recovered'].sum().reset_index()

    temp['No. of Deaths to 100 Confirmed Cases'] = round(temp['Deaths']/temp['Confirmed'], 3)*100

    temp['No. of Recovered to 100 Confirmed Cases'] = round(temp['Recovered']/temp['Confirmed'], 3)*100



    fig_1 = px.area(temp, x="Date", y="Confirmed", color_discrete_sequence = [act])

    fig_2 = px.area(temp, x="Date", y="Deaths", color_discrete_sequence = [dth])

    fig_3 = px.line(temp, x="Date", y="No. of Deaths to 100 Confirmed Cases",  color_discrete_sequence=['#333333'])

    fig_4 = px.line(temp, x="Date", y="No. of Recovered to 100 Confirmed Cases",  color_discrete_sequence=['#0f4c75'])



    # ------------------------------------------------



    # -------------------------------------------------------------------------------------------



    temp = covidData.loc[covidData['Country/Region']==country,:].groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']

    temp = temp.sum().diff().reset_index()



    mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



    temp.loc[mask, 'Confirmed'] = np.nan

    temp.loc[mask, 'Deaths'] = np.nan



    temp = temp.groupby('Date')['Confirmed'].sum().reset_index()



    fig_5 = px.bar(temp, x="Date", y="Confirmed", color_discrete_sequence = [act])

    fig_5.update_layout(xaxis_rangeslider_visible=True)



    # -------------------------------------------------------------------------------------------



    temp = covidData.loc[covidData['Country/Region']==country,:].groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths']

    temp = temp.sum().diff().reset_index()



    mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



    temp.loc[mask, 'Confirmed'] = np.nan

    temp.loc[mask, 'Deaths'] = np.nan



    temp = temp.groupby('Date')['Deaths'].sum().reset_index()



    fig_6 = px.bar(temp, x="Date", y="Deaths", color_discrete_sequence = [dth])

    fig_6.update_layout(xaxis_rangeslider_visible=True)







    # ==========================================================================================



    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.08, horizontal_spacing=0.1,

                        subplot_titles=('Confirmed cases', 'Deaths reported',

                                        'No. of new cases everyday','No. of new deaths everyday'))



    fig.add_trace(fig_1['data'][0], row=1, col=1)

    fig.add_trace(fig_2['data'][0], row=1, col=2)

    fig.add_trace(fig_5['data'][0], row=2, col=1)

    fig.add_trace(fig_6['data'][0], row=2, col=2)



    fig.update_layout(height=1200)

    fig.show()