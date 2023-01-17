from IPython.display import Image

Image(filename='/kaggle/input/worldcoronav/jobs-in-the-time-of-covid-19.jpg', width="800", height='50')
# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

%matplotlib inline

import seaborn as sns

import plotly as py

# py.offline.init_notebook_mode(connected = True)

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots

import folium

import math

import random

from datetime import timedelta

import warnings

warnings.filterwarnings('ignore')

# Color pallatte

cnf = '#393e46'

dth = '#ff2e63'

rec = '#21bf73'

act = '#fe9801'



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
country_day_wise = pd.read_csv("/kaggle/input/covid-19/country_daywise.csv", parse_dates = ['Date'])

country_wise = pd.read_csv("/kaggle/input/covid-19/countrywise.csv")

day_wise = pd.read_csv("/kaggle/input/covid-19/daywise.csv", parse_dates = ['Date'])

covid_19 = pd.read_csv("/kaggle/input/covid-19/covid_19_data_cleaned.csv", parse_dates = ['Date'])

covid_19.head(5)

# , parse_dates = ['Date']
covid_19.isnull().sum()
covid_19['Province/State'].value_counts()
# Filling missing values.

covid_19['Province/State'] = covid_19['Province/State'].fillna("")

covid_19.head()
Confirmed = covid_19.groupby('Date').sum()['Confirmed'].reset_index()

Recovered = covid_19.groupby('Date').sum()['Recovered'].reset_index()

Deaths = covid_19.groupby('Date').sum()['Deaths'].reset_index()

Active = covid_19.groupby('Date').sum()['Active'].reset_index()
covid_19.info()
covid_19.query('Country == "US"')
covid_19.query('Country == "Afghanistan"')
Confirmed.tail()
Recovered.tail()
Active.tail()
Deaths.tail()
fig = go.Figure()

fig.add_trace(go.Scatter(x = Confirmed['Date'], y = Confirmed['Confirmed'], mode = 'lines+markers', name = 'Confirmed Cases', line = dict(color='Orange')))

fig.add_trace(go.Scatter(x = Recovered['Date'], y = Recovered['Recovered'], mode = 'lines+markers', name = 'Recovred Cases', line = dict(color='Green')))

fig.add_trace(go.Scatter(x = Active['Date'], y = Active['Active'], mode = 'lines+markers', name = 'Active Cases', line = dict(color='blue')))

fig.add_trace(go.Scatter(x = Deaths['Date'], y = Deaths['Deaths'], mode = 'lines+markers', name = 'Deaths Cases', line = dict(color='Red')))

fig.update_layout(title='Worldwide Covid 19 Casess', xaxis_tickfont_size = 14, yaxis = dict(title = 'Number of Cases'))

fig.show()

covid_19.info()
# Change Date foramt to string format

covid_19['Date']  = covid_19['Date'].astype(str)

covid_19.info()
# Use plotly Express

fig = px.density_mapbox(covid_19, lat = 'Lat', lon = 'Long', hover_name = 'Country', hover_data = ['Confirmed', 'Recovered', 'Deaths'], animation_frame = 'Date', color_continuous_scale = 'Portland', radius = 7, zoom = 0, height=700)

fig.update_layout(title = 'Worldwide Covid_19 Cases with Time Laps')

fig.update_layout(mapbox_style = 'open-street-map', mapbox_center_lon = 0)

fig.show()
# Change string into Datetime format

covid_19['Date'] = pd.to_datetime(covid_19['Date'])

covid_19.info()
# Ships

# ==================

# Find out all Grand Princess

ship_rows = covid_19['Province/State'].str.contains('Grand Princess') | covid_19['Province/State'].str.contains('Diamond Princess') | covid_19['Country'].str.contains('Grand Princess') | covid_19['Country'].str.contains('Diamond Princess') | covid_19['Country'].str.contains('MS Zaandam') 

ship = covid_19[ship_rows]



covid_19 = covid_19[~ship_rows]
ship_latest = ship[ship['Date'] == max(ship['Date'])]

ship_latest
ship_latest.style.background_gradient(cmap = 'Pastel1_r')
temp = covid_19.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)



tm = temp.melt(id_vars = 'Date', value_vars = ['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path = ['variable'],values = 'value', height=250, width = 800, color_discrete_sequence=[act, rec, dth])



fig.data[0].textinfo = 'label+text+value'

fig.show()
# temp = covid_19.groupby('Date').sum()

temp = covid_19.groupby('Date')['Recovered', 'Deaths','Active'].sum().reset_index()

temp = temp.melt(id_vars = 'Date', value_vars = ['Recovered', 'Deaths','Active'], var_name = 'Case', value_name = 'Count')



fig = px.area(temp, x='Date', y='Count', color='Case', height=600, title='Cases over time', color_discrete_sequence=[act, rec, dth])

fig.update_layout(xaxis_rangeslider_visible = True)

fig.show()
# World wide map Cases on Folium Maps





temp = covid_19[covid_19['Date']==max(covid_19['Date'])]  # Latest Data Show

m = folium.Map(location=[0,0], tiles='cartodbpositron', min_zoom = 1, max_zoom = 4, zoom_start = 1)

for i in range(0, len(temp)):

    folium.Circle(location= [temp.iloc[i]['Lat'],temp.iloc[i]['Long']], color = 'crimson', fill = 'crimson',

                 tooltip =  '<li><bold> Country: ' + str(temp.iloc[i]['Country'])+

                            '<li><bold> Province: ' + str(temp.iloc[i]['Province/State'])+

                            '<li><bold> Confirmed: ' + str(temp.iloc[i]['Confirmed'])+

                            '<li><bold> Deaths: ' + str(temp.iloc[i]['Deaths']),

                 radius = int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)

m



country_day_wise.head(5)
fig = px.choropleth(country_day_wise, locations = 'Country', locationmode = 'country names', color = np.log(country_day_wise['Confirmed']),

                   hover_name = 'Country', animation_frame = country_day_wise['Date'].dt.strftime('%Y-%m-%d'),

                   title = 'Cases Over Time', color_continuous_scale = px.colors.sequential.Inferno)



fig.update(layout_coloraxis_showscale = True)

fig.show()
day_wise.head()
fig_c = px.bar(day_wise, x = 'Date', y='Confirmed', color_discrete_sequence=[act])

fig_d = px.bar(day_wise, x = 'Date', y='Deaths', color_discrete_sequence=[dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                  subplot_titles=('Confirmed Cases', 'Death Cases'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.update_layout(height=400)

fig.show()
fig_c = px.choropleth(country_wise, locations='Country', locationmode='country names',

                     color = np.log(country_wise['Confirmed']), hover_name = 'Country',

                     hover_data = ['Confirmed'])

temp = country_wise[country_wise['Deaths']>0]

fig_d = px.choropleth(temp, locations='Country', locationmode='country names',

                     color = np.log(temp['Deaths']), hover_name = 'Country',

                     hover_data = ['Deaths'])



fig = make_subplots(rows = 1, cols=2, subplot_titles=['Confirmed','Deaths'],

                  specs=[[{'type': 'choropleth'},{'type': 'choropleth'} ]])

fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.update(layout_coloraxis_showscale=False)

fig.show()
fig1 = px.line(day_wise, x='Date', y='Deaths / 100 Cases',color_discrete_sequence=[dth])

fig2 = px.line(day_wise, x='Date', y='Recovered / 100 Cases',color_discrete_sequence=[rec])

fig3 = px.line(day_wise, x='Date', y='Deaths / 100 Recovered',color_discrete_sequence=['aqua'])



fig = make_subplots(rows=1, cols=3, shared_xaxes=False,

                   subplot_titles=("Deaths / 100 Cases", 'Recovered / 100 Cases','Deaths / 100 Recovered'))

fig.add_trace(fig1['data'][0], row=1,col=1)

fig.add_trace(fig2['data'][0], row=1,col=2)

fig.add_trace(fig3['data'][0], row=1,col=3)



fig.update_layout(height=400)

fig.show()
fig_c = px.bar(day_wise, x='Date', y='Confirmed', color_discrete_sequence=[act])

fig_d = px.bar(day_wise, x='Date', y='No. of Countries', color_discrete_sequence=[dth])



fig = make_subplots(rows=1, cols=2, shared_xaxes=False, horizontal_spacing=0.1,

                   subplot_titles=("Number of new Cases per Day", 'No. of Countries'))

fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.show()
country_wise.columns
top = 15



fig_c = px.bar(country_wise.sort_values('Confirmed').tail(top), x='Confirmed', y='Country',

              text = 'Confirmed', orientation='h', color_discrete_sequence=[cnf])

fig_d = px.bar(country_wise.sort_values('Deaths').tail(top), x='Deaths', y='Country',

              text = 'Deaths', orientation='h', color_discrete_sequence=[dth])



fig_a = px.bar(country_wise.sort_values('Active').tail(top), x='Active', y='Country',

              text = 'Active', orientation='h', color_discrete_sequence=['#434343'])

fig_r = px.bar(country_wise.sort_values('Recovered').tail(top), x='Recovered', y='Country',

              text = 'Recovered', orientation='h', color_discrete_sequence=[rec])



# Plot Deaths / 100 Cases in world



fig_dc = px.bar(country_wise.sort_values('Deaths / 100 Cases').tail(top), x='Deaths / 100 Cases', y='Country',

              text = 'Deaths / 100 Cases', orientation='h', color_discrete_sequence=['#f84351'])



# Plot Recovered / 100 Cases in world



fig_rc = px.bar(country_wise.sort_values('Recovered / 100 Cases').tail(top), x='Recovered / 100 Cases', y='Country',

              text = 'Recovered / 100 Cases', orientation='h', color_discrete_sequence=['#a45998'])



# New Cases  per milion people



fig_nc = px.bar(country_wise.sort_values('New Cases').tail(top), x='New Cases', y='Country',

              text = 'New Cases', orientation='h', color_discrete_sequence=['#f04341'])



temp = country_wise[country_wise['Population']>1000000]

fig_p = px.bar(temp.sort_values('Cases / Million People').tail(top), x='Cases / Million People', y='Country',

              text = 'Cases / Million People', orientation='h', color_discrete_sequence=['#b40398'])



# New Cases  per One week Changes people



fig_wc = px.bar(country_wise.sort_values('1 week change').tail(top), x='1 week change', y='Country',

              text = '1 week change', orientation='h', color_discrete_sequence=['#f04554'])



temp = country_wise[country_wise['Confirmed']>100]

fig_wi = px.bar(temp.sort_values('1 week % increase').tail(top), x='1 week % increase', y='Country',

              text = '1 week % increase', orientation='h', color_discrete_sequence=['#b08692'])



fig = make_subplots(rows=5, cols=2, shared_xaxes=False, horizontal_spacing=0.2,

                    vertical_spacing=.05,

                    subplot_titles=('Confirmed Cases', 'Deaths Reported', "Recovered Cases",

                                    'Active Cases','Deaths / 100 Cases','Recovered / 100 Cases',

                                   'New Cases','Cases / Million People','1 week change','1 week % increase'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)



fig.add_trace(fig_r['data'][0], row=2, col=1)

fig.add_trace(fig_a['data'][0], row=2, col=2)



fig.add_trace(fig_dc['data'][0], row=3, col=1)

fig.add_trace(fig_rc['data'][0], row=3, col=2)



fig.add_trace(fig_nc['data'][0], row=4, col=1)

fig.add_trace(fig_p['data'][0], row=4, col=2)



fig.add_trace(fig_wc['data'][0], row=5, col=1)

fig.add_trace(fig_wi['data'][0], row=5, col=2)



fig.update_layout(height=4000)

fig.show()
# country_wise.sor_values['Deaths', ascending=False].iloc[:15, :]

top = 15

fig = px.scatter(country_wise.sort_values('Deaths', ascending=False).head(top),

                x = 'Confirmed', y='Deaths', color='Country', size='Confirmed', height=700,

                text = 'Country', log_x = True, title='Deaths vs Confirmed Cases(Caes are on log10 Scale)')

fig.update_traces(textposition = 'top center')

fig.update_layout(showlegend = False)

fig.update_layout(xaxis_rangeslider_visible= True)

fig.show()
country_day_wise.head(2)
fig = px.bar(country_day_wise, x = 'Date', y='Confirmed', color='Country', height=600,

            title='Confirmed Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig = px.bar(country_day_wise, x = 'Date', y='Deaths', color='Country', height=600,

            title='Deaths Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
country_day_wise.head(2)
fig = px.bar(country_day_wise, x = 'Date', y='Recovered', color='Country', height=600,

            title='Recovered Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
fig = px.bar(country_day_wise, x = 'Date', y='New Cases', color='Country', height=600,

            title='New Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
# Confirmed Cases

fig = px.line(country_day_wise, x = 'Date', y='Confirmed', color='Country', height=600,

              title='Confirmed Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()



# Death Cases



fig = px.line(country_day_wise, x = 'Date', y='Deaths', color='Country', height=600,

              title='Deaths Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()



# Recovered Cases



fig = px.line(country_day_wise, x = 'Date', y='Recovered', color='Country', height=600,

              title='Recovered Cases',color_discrete_sequence=px.colors.cyclical.mygbm)

fig.show()
gt_100 = country_day_wise[country_day_wise['Confirmed']<100]

gt_100
gt_100 = country_day_wise[country_day_wise['Confirmed']>100]['Country'].unique()

temp = covid_19[covid_19['Country'].isin(gt_100)]



temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>100]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country', 'Min Date']



from_100th_case = pd.merge(temp, min_date, on='Country')

from_100th_case['N days'] = (from_100th_case['Date'] - from_100th_case['Min Date']).dt.days

fig = px.line(from_100th_case, x = 'N days', y='Confirmed', color='Country', title='N days from 100 cases',

             height=600)

fig.show()
gt_1000 = country_day_wise[country_day_wise['Confirmed']>1000]['Country'].unique()

temp = covid_19[covid_19['Country'].isin(gt_1000)]



temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>1000]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country', 'Min Date']



from_1000th_case = pd.merge(temp, min_date, on='Country')

from_1000th_case['N days'] = (from_1000th_case['Date'] - from_1000th_case['Min Date']).dt.days

fig = px.line(from_1000th_case, x = 'N days', y='Confirmed', color='Country', title='N days from 1000 cases',

             height=600)

fig.show()
gt_10000 = country_day_wise[country_day_wise['Confirmed']>10000]['Country'].unique()

temp = covid_19[covid_19['Country'].isin(gt_10000)]



temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>10000]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country', 'Min Date']



from_10000th_case = pd.merge(temp, min_date, on='Country')

from_10000th_case['N days'] = (from_10000th_case['Date'] - from_10000th_case['Min Date']).dt.days

fig = px.line(from_10000th_case, x = 'N days', y='Confirmed', color='Country', title='N days from 10000 cases',

             height=600)

fig.show()
gt_100000 = country_day_wise[country_day_wise['Confirmed']>100000]['Country'].unique()

temp = covid_19[covid_19['Country'].isin(gt_100000)]



temp = temp.groupby(['Country', 'Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>100000]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country', 'Min Date']



from_100000th_case = pd.merge(temp, min_date, on='Country')

from_100000th_case['N days'] = (from_100000th_case['Date'] - from_100000th_case['Min Date']).dt.days

fig = px.line(from_100000th_case, x = 'N days', y='Confirmed', color='Country', title='N days from 100000 cases',

             height=600)

fig.show()
covid_19.head(2)
full_latest = covid_19[covid_19['Date'] == max(covid_19['Date'])]



fig = px.treemap(full_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True),

                path = ['Country', 'Province/State'], values='Confirmed', height=700,

                title='Number of Confirmed Cases',

                color_discrete_sequence=px.colors.qualitative.Dark2)



fig.data[0].textinfo = 'label+text+value'

fig.show()
full_latest = covid_19[covid_19['Date'] == max(covid_19['Date'])]



fig = px.treemap(full_latest.sort_values(by='Confirmed', ascending=False).reset_index(drop=True),

                path = ['Country', 'Province/State'], values='Deaths', height=700,

                title='Number of Deaths Cases',

                color_discrete_sequence=px.colors.qualitative.Dark2)



fig.data[0].textinfo = 'label+text+value'

fig.show()