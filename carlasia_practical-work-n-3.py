# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
! pip install us
# essential libraries

import math

import random

from datetime import timedelta

from urllib.request import urlopen

import json



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



import folium





# color pallette

cnf, dth, rec, act = '#393e46', '#ff2e63', '#21bf73', '#fe9801' 



# converter

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()   



# hide warnings

import warnings

warnings.filterwarnings('ignore')



from plotly.offline import plot, iplot, init_notebook_mode

init_notebook_mode(connected=True)
full_table = pd.read_csv('../input/praticalwork/covid_19_clean_complete.csv', parse_dates=['Date'])

full_table
# Ship

# ====

# ship rows

ship_rows = full_table['Province/State'].str.contains('Grand Princess') | full_table['Province/State'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('Diamond Princess') | full_table['Country/Region'].str.contains('MS Zaandam')



# ship

ship = full_table[ship_rows]



# full table 

full_table = full_table[~(ship_rows)]



# Latest cases from the ships

ship_latest = ship[ship['Date']==max(ship['Date'])]
# Cleaning data

# =============



# fixing Country values

full_table.loc[full_table['Province/State']=='Greenland', 'Country/Region'] = 'Greenland'



# Active Case = confirmed - deaths - recovered

full_table['Active'] = full_table['Confirmed'] - full_table['Deaths'] - full_table['Recovered']



# replacing Mainland china with just China

full_table['Country/Region'] = full_table['Country/Region'].replace('Mainland China', 'China')



# filling missing values 

full_table[['Province/State']] = full_table[['Province/State']].fillna('')

full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']] = full_table[['Confirmed', 'Deaths', 'Recovered', 'Active']].fillna(0)



# fixing datatypes

full_table['Recovered'] = full_table['Recovered'].astype(int)



full_table.to_csv('/root/full_table.csv', index = False)

full_table.head()
# Grouped by day, country

# =======================



full_grouped = full_table.groupby(['Date', 'Country/Region'])['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()



# new cases ======================================================

temp = full_grouped.groupby(['Country/Region', 'Date', ])['Confirmed', 'Deaths', 'Recovered']

temp = temp.sum().diff().reset_index()



mask = temp['Country/Region'] != temp['Country/Region'].shift(1)



temp.loc[mask, 'Confirmed'] = np.nan

temp.loc[mask, 'Deaths'] = np.nan

temp.loc[mask, 'Recovered'] = np.nan



# renaming columns

temp.columns = ['Country/Region', 'Date', 'New cases', 'New deaths', 'New recovered']

# =================================================================



# merging new values

full_grouped = pd.merge(full_grouped, temp, on=['Country/Region', 'Date'])



# filling na with 0

full_grouped = full_grouped.fillna(0)



# fixing data types

cols = ['New cases', 'New deaths', 'New recovered']

full_grouped[cols] = full_grouped[cols].astype('int')

full_grouped['New cases'] = full_grouped['New cases'].apply(lambda x: 0 if x<0 else x)

full_grouped.head()

# Day wise

# ========



# table

day_wise = full_grouped.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases', 'New deaths'].sum().reset_index()



# number cases per 100 cases

day_wise['Deaths / 100 Cases'] = round((day_wise['Deaths']/day_wise['Confirmed'])*100, 2)

day_wise['Recovered / 100 Cases'] = round((day_wise['Recovered']/day_wise['Confirmed'])*100, 2)

day_wise['Deaths / 100 Recovered'] = round((day_wise['Deaths']/day_wise['Recovered'])*100, 2)



# no. of countries

day_wise['No. of countries'] = full_grouped[full_grouped['Confirmed']!=0].groupby('Date')['Country/Region'].unique().apply(len).values



# fillna by 0

cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']

day_wise[cols] = day_wise[cols].fillna(0)



day_wise.head()
# Country wise

# ============



# getting latest values

country_wise = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)



# group by country

country_wise = country_wise.groupby('Country/Region')['Confirmed', 'Deaths', 'Recovered', 'Active', 'New cases'].sum().reset_index()



# per 100 cases

country_wise['Deaths / 100 Cases'] = round((country_wise['Deaths']/country_wise['Confirmed'])*100, 2)

country_wise['Recovered / 100 Cases'] = round((country_wise['Recovered']/country_wise['Confirmed'])*100, 2)

country_wise['Deaths / 100 Recovered'] = round((country_wise['Deaths']/country_wise['Recovered'])*100, 2)



cols = ['Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered']

country_wise[cols] = country_wise[cols].fillna(0)



country_wise
# load population dataset

pop = pd.read_csv("../input/praticalwork/population_clean.csv")



# select only population

pop1 = pop.iloc[:,0]

pop2 = pop.iloc[:,-1]

pop = pd.concat([pop1,pop2], axis = 1)

pop

# rename column names

pop.columns = ['Country/Region', 'Population']



# merged data

country_wise = country_wise.merge(pop, on='Country/Region')

a = country_wise['Confirmed'] / country_wise['Population']



# update population

cols = ['Burma', 'Congo (Brazzaville)', 'Congo (Kinshasa)', "Cote d'Ivoire", 'Czechia', 

        'Kosovo', 'Saint Kitts and Nevis', 'Saint Vincent and the Grenadines', 

        'Taiwan*', 'US', 'West Bank and Gaza', 'Sao Tome and Principe','Eritrea']

pops = [54409800, 89561403, 5518087, 26378274, 10708981, 1793000, 

        53109, 110854, 23806638, 330541757, 4543126, 219159,3214657]

for c, p in zip(cols, pops):

    country_wise.loc[country_wise['Country/Region']== c, 'Population'] = p



# missing values

# country_wise.isna().sum()

# country_wise[country_wise['Population'].isna()]['Country/Region'].tolist()



# Cases per population

country_wise['Cases / Million People'] = round((country_wise['Confirmed'] / country_wise['Population']) * 1000000)



country_wise.head()



country_wise.to_csv('/root/real_table.csv', index = False)



country_wise.head()

today = full_grouped[full_grouped['Date']==max(full_grouped['Date'])].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]

last_week = full_grouped[full_grouped['Date']==max(full_grouped['Date'])-timedelta(days=7)].reset_index(drop=True).drop('Date', axis=1)[['Country/Region', 'Confirmed']]



temp = pd.merge(today, last_week, on='Country/Region', suffixes=(' today', ' last week'))



# temp = temp[['Country/Region', 'Confirmed last week']]

temp['1 week change'] = temp['Confirmed today'] - temp['Confirmed last week']



temp = temp[['Country/Region', 'Confirmed last week', '1 week change']]



country_wise = pd.merge(country_wise, temp, on='Country/Region')



country_wise['1 week % increase'] = round(country_wise['1 week change']/country_wise['Confirmed last week']*100, 2)



country_wise.head()

country_wise[country_wise['Cases / Million People'].isna()]
temp = full_table.groupby('Date')['Confirmed', 'Deaths', 'Recovered', 'Active'].sum().reset_index()

temp = temp[temp['Date']==max(temp['Date'])].reset_index(drop=True)



tm = temp.melt(id_vars="Date", value_vars=['Active', 'Deaths', 'Recovered'])

fig = px.treemap(tm, path=["variable"], values="value", height=225, width=1200,

                 color_discrete_sequence=[act, rec, dth])

fig.data[0].textinfo = 'label+text+value'

fig.show()
temp = full_table.groupby('Date')['Recovered', 'Deaths', 'Active'].sum().reset_index()

temp = temp.melt(id_vars="Date", value_vars=['Recovered', 'Deaths', 'Active'],

                 var_name='Case', value_name='Count')

temp.head()



fig = px.area(temp, x="Date", y="Count", color='Case', height=600,

             title='Cases over time', color_discrete_sequence = [rec, dth, act])

fig.update_layout(xaxis_rangeslider_visible=True)

fig.show()
# World wide



temp = full_table[full_table['Date'] == max(full_table['Date'])]



m = folium.Map(location=[0, 0], tiles='cartodbpositron',

               min_zoom=1, max_zoom=4, zoom_start=1)



for i in range(0, len(temp)):

    folium.Circle(

        location=[temp.iloc[i]['Lat'], temp.iloc[i]['Long']],

        color='crimson', fill='crimson',

        tooltip =   '<li><bold>Country : '+str(temp.iloc[i]['Country/Region'])+

                    '<li><bold>Province : '+str(temp.iloc[i]['Province/State'])+

                    '<li><bold>Confirmed : '+str(temp.iloc[i]['Confirmed'])+'<li><bold>Deaths : '+str(temp.iloc[i]['Deaths']),

        radius=int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)

m
# Over the time



fig = px.choropleth(full_grouped, locations="Country/Region", locationmode='country names', color=np.log(full_grouped["Confirmed"]), 

                    hover_name="Country/Region", animation_frame=full_grouped["Date"].dt.strftime('%Y-%m-%d'),

                    title='Cases over time', color_continuous_scale=px.colors.sequential.Purp)

fig.update(layout_coloraxis_showscale=False)

fig.show()
fig_c = px.bar(day_wise, x="Date", y="Confirmed", color_discrete_sequence = [act])

fig_d = px.bar(day_wise, x="Date", y="Deaths", color_discrete_sequence = [dth])

fig_r = px.bar(day_wise, x="Date", y="Recovered", color_discrete_sequence = [rec])



fig = make_subplots(rows=1, cols=3, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('Confirmed cases', 'Deaths reported', 

                                    'Recovered reported'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.add_trace(fig_r['data'][0], row=1, col=3)





fig.update_layout(height=480)

fig.show()
# ===============================



fig_1 = px.line(day_wise, x="Date", y="Deaths / 100 Cases", color_discrete_sequence = [dth])

fig_2 = px.line(day_wise, x="Date", y="Recovered / 100 Cases", color_discrete_sequence = [rec])

fig_3 = px.line(day_wise, x="Date", y="Deaths / 100 Recovered", color_discrete_sequence = ['#333333'])



fig = make_subplots(rows=1, cols=3, shared_xaxes=False, 

                    subplot_titles=('Deaths / 100 Cases', 'Recovered / 100 Cases', 'Deaths / 100 Recovered'))



fig.add_trace(fig_1['data'][0], row=1, col=1)

fig.add_trace(fig_2['data'][0], row=1, col=2)

fig.add_trace(fig_3['data'][0], row=1, col=3)



fig.update_layout(height=480)

fig.show()



# ===================================
fig_c = px.bar(day_wise, x="Date", y="New cases", color_discrete_sequence = [act])

fig_n = px.bar(day_wise, x="Date", y="New deaths", color_discrete_sequence = [dth])

fig_d = px.bar(day_wise, x="Date", y="No. of countries", color_discrete_sequence = ['#333333'])



fig = make_subplots(rows=1, cols=3, shared_xaxes=False, horizontal_spacing=0.1,

                    subplot_titles=('No. of new cases everyday',

                                    'No. of new deaths everyday',

                                    'No. of countries'))



fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_c['data'][0], row=1, col=2)

fig.add_trace(fig_d['data'][0], row=1, col=3)



fig.update_layout(height=480)

fig.show()
# confirmed - deaths

fig_c = px.bar(country_wise.sort_values('Confirmed').tail(15), x="Confirmed", y="Country/Region", 

               text='Confirmed', orientation='h', color_discrete_sequence = [act])

fig_d = px.bar(country_wise.sort_values('Deaths').tail(15), x="Deaths", y="Country/Region", 

               text='Deaths', orientation='h', color_discrete_sequence = [dth])



# recovered - active

fig_r = px.bar(country_wise.sort_values('Recovered').tail(15), x="Recovered", y="Country/Region", 

               text='Recovered', orientation='h', color_discrete_sequence = [rec])

fig_a = px.bar(country_wise.sort_values('Active').tail(15), x="Active", y="Country/Region", 

               text='Active', orientation='h', color_discrete_sequence = ['#333333'])

# death - recoverd / 100 cases

fig_dc = px.bar(country_wise.sort_values('Deaths / 100 Cases').tail(15), x="Deaths / 100 Cases", y="Country/Region", 

               text='Deaths / 100 Cases', orientation='h', color_discrete_sequence = ['#f38181'])

fig_rc = px.bar(country_wise.sort_values('Recovered / 100 Cases').tail(15), x="Recovered / 100 Cases", y="Country/Region", 

               text='Recovered / 100 Cases', orientation='h', color_discrete_sequence = ['#a3de83'])



# new cases - cases per million people

fig_nc = px.bar(country_wise.sort_values('New cases').tail(15), x="New cases", y="Country/Region", 

               text='New cases', orientation='h', color_discrete_sequence = ['#c61951'])

temp = country_wise[country_wise['Population']>1000000]

fig_p = px.bar(temp.sort_values('Cases / Million People').tail(15), x="Cases / Million People", y="Country/Region", 

               text='Cases / Million People', orientation='h', color_discrete_sequence = ['#741938'])

# week change, percent increase

fig_wc = px.bar(country_wise.sort_values('1 week change').tail(15), x="1 week change", y="Country/Region", 

               text='1 week change', orientation='h', color_discrete_sequence = ['#004a7c'])

temp = country_wise[country_wise['Confirmed']>100]

fig_pi = px.bar(temp.sort_values('1 week % increase').tail(15), x="1 week % increase", y="Country/Region", 

               text='1 week % increase', orientation='h', color_discrete_sequence = ['#005691'], 

                hover_data=['Confirmed last week', 'Confirmed'])





# plot

fig = make_subplots(rows=5, cols=2, shared_xaxes=False, horizontal_spacing=0.14, vertical_spacing=0.08,

                    subplot_titles=('Confirmed cases', 'Deaths reported', 'Recovered', 'Active cases', 

                                    'Deaths / 100 cases', 'Recovered / 100 cases', 'New cases', 

                                    'Cases / Million People', '1 week increase', '1 week % increase'))

fig.add_trace(fig_c['data'][0], row=1, col=1)

fig.add_trace(fig_d['data'][0], row=1, col=2)

fig.add_trace(fig_r['data'][0], row=2, col=1)

fig.add_trace(fig_a['data'][0], row=2, col=2)



fig.add_trace(fig_dc['data'][0], row=3, col=1)

fig.add_trace(fig_rc['data'][0], row=3, col=2)

fig.add_trace(fig_nc['data'][0], row=4, col=1)

fig.add_trace(fig_p['data'][0], row=4, col=2)



fig.add_trace(fig_wc['data'][0], row=5, col=1)

fig.add_trace(fig_pi['data'][0], row=5, col=2)





fig.update_layout(height=3000)


temp = full_table.groupby('Date')['Confirmed'].sum()

temp = temp.diff()

temp.head()



plt.figure(figsize=(20, 5))

ax = calmap.yearplot(temp, fillcolor='white', cmap='Reds', linewidth=0.5)

ax