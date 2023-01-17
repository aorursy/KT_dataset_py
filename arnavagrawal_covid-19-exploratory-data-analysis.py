!pip install folium plotly
# imports

import plotly.express as px

import plotly.graph_objects as go

import plotly.figure_factory as ff

from plotly.subplots import make_subplots



import folium



import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline



import math

import random

from datetime import timedelta



import warnings

warnings.filterwarnings('ignore')



# color pallette

cnf = '#393e46'

dth = '#ff2e63'

rec = '#21bf73'

act = '#fe9801'
import plotly as py

py.offline.init_notebook_mode(connected = True)
import os

try:

    os.system("rm -rf Covid-19-Preprocessed-Dataset")

except:

    print("File does not exist")
!git clone https://github.com/laxmimerit/Covid-19-Preprocessed-Dataset.git
df = pd.read_csv("Covid-19-Preprocessed-Dataset/preprocessed/covid_19_data_cleaned.csv", parse_dates=['Date'])

country_daywise = pd.read_csv("Covid-19-Preprocessed-Dataset/preprocessed/country_daywise.csv", parse_dates=['Date'])

countrywise = pd.read_csv("Covid-19-Preprocessed-Dataset/preprocessed/countrywise.csv")

daywise = pd.read_csv("Covid-19-Preprocessed-Dataset/preprocessed/daywise.csv", parse_dates=['Date'])
df['Province/State'] = df['Province/State'].fillna("")

df
country_daywise
countrywise
daywise
confirmed = df.groupby('Date').sum()['Confirmed'].reset_index()

recovered = df.groupby('Date').sum()['Recovered'].reset_index()

deaths = df.groupby('Date').sum()['Deaths'].reset_index()
df.isnull().sum()
df.info()
df.describe()
df.query('Country == "India"')
countrywise.query('Country == "India"')
country_daywise.query('Country == "India"')
confirmed.tail()
recovered.tail()
deaths.tail()
fig = go.Figure()

fig.add_trace(go.Scatter(x = confirmed['Date'], y = confirmed['Confirmed'],

                         mode = 'lines+markers', name = 'Confirmed', 

                         line = dict(color = "Red", width = 2)))

fig.add_trace(go.Scatter(x = recovered['Date'], y = recovered['Recovered'],

                         mode = 'lines+markers', name = 'Recovered', 

                         line = dict(color = "Green", width = 2)))

fig.add_trace(go.Scatter(x = deaths['Date'], y = deaths['Deaths'],

                         mode = 'lines+markers', name = 'Deaths', 

                         line = dict(color = "Grey", width = 2)))

fig.update_layout(title = 'Worldwide Covid-19 Cases', xaxis_tickfont_size = 14, 

                 yaxis = dict(title = 'Number of Cases'))

fig.show()
df.info()
df['Date'] = df['Date'].astype(str)
df.info()
fig = px.density_mapbox(df, lat = 'Lat', lon = 'Long', 

                        hover_name = 'Country', hover_data = ['Confirmed','Recovered','Active','Deaths'],

                        animation_frame = 'Date', color_continuous_scale = 'Portland', radius = 7,

                        zoom = 0, height = 700)

fig.update_layout(title = 'Worldwide Covid-19 Cases with Time Lapse')

fig.update_layout(mapbox_style = 'open-street-map', mapbox_center_lon = 0)

fig.show()
temp = df.groupby('Date')['Confirmed','Recovered','Active','Deaths'].sum().reset_index()

temp = temp[temp['Date'] == max(temp['Date'])].reset_index(drop = True)

temp
tm = temp.melt(id_vars = 'Date', value_vars = ['Confirmed','Active','Recovered','Deaths'])

fig = px.treemap(tm, path = ['variable'], values = 'value', height = 250, width = 1000, 

                 color_discrete_sequence = [cnf,rec,act,dth])

fig.data[0].textinfo = 'label+text+value'

fig.show()
temp1 = df.groupby('Date')['Recovered','Deaths','Active'].sum().reset_index()

temp1 = temp1.melt(id_vars = 'Date', value_vars = ['Recovered','Deaths','Active'], 

                   var_name = 'Case', value_name = 'Count')

fig = px.area(temp1, x = 'Date', y = 'Count', color = 'Case', height = 600, title = 'Cases over Time', 

              color_discrete_sequence = [rec,dth,act])

fig.update_layout(xaxis_rangeslider_visible = True)

fig.show()
temp = df[df['Date'] == max(df['Date'])]

temp
m = folium.Map(location = [0,0], tiles = 'cartodbpositron', min_zoom = 1, max_zoom = 4, zoom_start = 1)

for i in range(0,len(temp)):

    folium.Circle(location = [temp.iloc[i]['Lat'], temp.iloc[i]['Long']], color = 'crimson', fill = 'crimson',

                  tooltip = '<li><bold> Country: ' + str(temp.iloc[i]['Country']) + 

                     '<li><bold> Province: ' + str(temp.iloc[i]['Province/State']) + 

                     '<li><bold> Confirmed: ' + str(temp.iloc[i]['Confirmed']) + 

                     '<li><bold> Deaths: ' + str(temp.iloc[i]['Deaths']),

                  radius = int(temp.iloc[i]['Confirmed'])**0.5).add_to(m)

m
fig = px.choropleth(country_daywise, locations = 'Country', locationmode = 'country names',

                    color = country_daywise['Confirmed'], hover_name = 'Country', hover_data = ['Confirmed'],

                    animation_frame = country_daywise['Date'].dt.strftime('%Y-%m-%d'), title='Cases over time', 

                    color_continuous_scale = px.colors.sequential.Inferno)

fig.update(layout_coloraxis_showscale = True)

fig.show()
daywise.head()
fig_c = px.bar(daywise, x = 'Date', y = 'Confirmed', color_discrete_sequence = [act])

fig_d = px.bar(daywise, x = 'Date', y = 'Deaths', color_discrete_sequence = [dth])

fig = make_subplots(rows = 1, cols = 2, shared_xaxes = False, horizontal_spacing = 0.1, 

                    subplot_titles = ('Confirmed Cases', 'Death Cases'))

fig.add_trace(fig_c['data'][0], row = 1, col = 1)

fig.add_trace(fig_d['data'][0], row = 1, col = 2)

fig.update_layout(height = 480)

fig.show()
fig_c = px.choropleth(countrywise, locations = 'Country', locationmode = 'country names', 

                      color = np.log(countrywise['Confirmed']), hover_name = 'Country', 

                      hover_data = ['Confirmed'])

temp = countrywise[countrywise['Deaths']>0]

fig_d = px.choropleth(temp, locations = 'Country', locationmode = 'country names', 

                      color = np.log(temp['Deaths']), hover_name = 'Country', 

                      hover_data = ['Deaths'])

fig = make_subplots(rows = 1 ,cols = 2, subplot_titles = ('Confirmed Cases', 'Death Cases'),

                    specs = [[{'type':'choropleth'}, {'type':'choropleth'}]])

fig.add_trace(fig_c['data'][0], row = 1, col = 1)

fig.add_trace(fig_d['data'][0], row = 1, col = 2)

fig.update(layout_coloraxis_showscale = False)

fig.show()
fig1 = px.line(daywise, x = 'Date', y = 'Deaths / 100 Cases', color_discrete_sequence = [dth])

fig2 = px.line(daywise, x = 'Date', y = 'Recovered / 100 Cases', color_discrete_sequence = [rec])

fig3 = px.line(daywise, x = 'Date', y = 'Deaths / 100 Recovered', color_discrete_sequence = ['blue'])

fig = make_subplots(rows = 1, cols = 3, shared_xaxes = False,

                    subplot_titles = ('Deaths / 100 Cases','Recovered / 100 Cases', 'Deaths / 100 Recovered'))

fig.add_trace(fig1['data'][0], row = 1, col = 1)

fig.add_trace(fig2['data'][0], row = 1, col = 2)

fig.add_trace(fig3['data'][0], row = 1, col = 3)

fig.update_layout(height = 480)

fig.show()
fig_c = px.bar(daywise, x = 'Date', y = 'Confirmed', color_discrete_sequence = [act])

fig_d = px.bar(daywise, x = 'Date', y = 'No. of Countries', color_discrete_sequence = [dth])

fig = make_subplots(rows = 1, cols = 2, shared_xaxes = False, horizontal_spacing = 0.1, 

                    subplot_titles = ('No. of New Cases per Day','No. of Countries'))

fig.add_trace(fig_c['data'][0], row = 1, col = 1)

fig.add_trace(fig_d['data'][0], row = 1, col = 2)

fig.update_layout(height = 480)

fig.show()
countrywise.columns
top = 15

fig_c = px.bar(countrywise.sort_values('Confirmed').tail(top), x = 'Confirmed', y = 'Country', 

               text = 'Confirmed', orientation = 'h', color_discrete_sequence = [cnf])

fig_d = px.bar(countrywise.sort_values('Deaths').tail(top), x = 'Deaths', y = 'Country', 

               text = 'Deaths', orientation = 'h', color_discrete_sequence = [dth])



fig_a = px.bar(countrywise.sort_values('Active').tail(top), x = 'Active', y = 'Country', 

               text = 'Active', orientation = 'h', color_discrete_sequence = [act])

fig_r = px.bar(countrywise.sort_values('Recovered').tail(top), x = 'Recovered', y = 'Country', 

               text = 'Recovered', orientation = 'h', color_discrete_sequence = [rec])



fig_dc = px.bar(countrywise.sort_values('Deaths / 100 Cases').tail(top), x = 'Deaths / 100 Cases', y = 'Country', 

               text = 'Deaths / 100 Cases', orientation = 'h', color_discrete_sequence = ['#f84351'])

fig_rc = px.bar(countrywise.sort_values('Recovered / 100 Cases').tail(top), x = 'Recovered / 100 Cases', y = 'Country', 

               text = 'Recovered / 100 Cases', orientation = 'h', color_discrete_sequence = ['#a45398'])



fig_nc = px.bar(countrywise.sort_values('New Cases').tail(top), x = 'New Cases', y = 'Country', 

               text = 'New Cases', orientation = 'h', color_discrete_sequence = ['#f04341'])

temp = countrywise[countrywise['Population']>1000000]

fig_p = px.bar(temp.sort_values('Cases / Million People').tail(top), x = 'Cases / Million People', y = 'Country', 

               text = 'Cases / Million People', orientation = 'h', color_discrete_sequence = ['#b40398'])



fig_wc = px.bar(countrywise.sort_values('1 week change').tail(top), x = '1 week change', y = 'Country', 

               text = '1 week change', orientation = 'h', color_discrete_sequence = ['#c04041'])

temp = countrywise[countrywise['Confirmed']>100]

fig_wi = px.bar(temp.sort_values('1 week % increase').tail(top), x = '1 week % increase', y = 'Country', 

               text = '1 week % increase', orientation = 'h', color_discrete_sequence = ['#f00398'])



fig = make_subplots(rows = 5, cols = 2, shared_xaxes = False, horizontal_spacing = 0.14,

                    vertical_spacing = 0.1, subplot_titles = ('Confirmed Cases','Deaths Reported',

                                                              'Active Cases','Recovered Cases',

                                                             'Deaths / 100 Cases','Recovered / 100 Cases',

                                                              'New Cases','Cases / Million People',

                                                             '1 Week Change','1 Week % Increase'))

fig.add_trace(fig_c['data'][0], row = 1, col = 1)

fig.add_trace(fig_d['data'][0], row = 1, col = 2)



fig.add_trace(fig_a['data'][0], row = 2, col = 1)

fig.add_trace(fig_r['data'][0], row = 2, col = 2)



fig.add_trace(fig_dc['data'][0], row = 3, col = 1)

fig.add_trace(fig_rc['data'][0], row = 3, col = 2)



fig.add_trace(fig_nc['data'][0], row = 4, col = 1)

fig.add_trace(fig_p['data'][0], row = 4, col = 2)



fig.add_trace(fig_wc['data'][0], row = 5, col = 1)

fig.add_trace(fig_wi['data'][0], row = 5, col = 2)



fig.update_layout(height = 3000)

fig.show()
top = 15

fig = px.scatter(countrywise.sort_values('Deaths', ascending = False).head(top), x = 'Confirmed', y = 'Deaths', 

                 color = 'Country', size = 'Confirmed', height = 700, text = 'Country', log_x = True, log_y = True,

                 title = 'Deaths vs Confirmed Cases')

fig.update_traces(textposition = 'top center')

fig.update_layout(showlegend = False)

fig.update_layout(xaxis_rangeslider_visible = True)

fig.show()
fig = px.bar(country_daywise, x = 'Date', y = 'Confirmed', color = 'Country', height = 600,

             title = 'Confirmed', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.bar(country_daywise, x = 'Date', y = 'Deaths', color = 'Country', height = 600,

             title = 'Deaths', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.bar(country_daywise, x = 'Date', y = 'Recovered', color = 'Country', height = 600,

             title = 'Recovered', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.bar(country_daywise, x = 'Date', y = 'New Cases', color = 'Country', height = 600,

             title = 'New Cases', color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.line(country_daywise, x ='Date', y = 'Confirmed', color = 'Country', height = 600, title = 'Confirmed',

              color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.line(country_daywise, x ='Date', y = 'Deaths', color = 'Country', height = 600, title = 'Deaths',

              color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
fig = px.line(country_daywise, x ='Date', y = 'Recovered', color = 'Country', height = 600, title = 'Recovered',

              color_discrete_sequence = px.colors.cyclical.mygbm)

fig.show()
gt_100 = country_daywise[country_daywise['Confirmed']>100]['Country'].unique()

temp = df[df['Country'].isin(gt_100)]

temp = temp.groupby(['Country','Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>100]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country','Min Date']



from_100th_case = pd.merge(temp, min_date, on = 'Country')

from_100th_case['N days'] = (pd.to_datetime(from_100th_case['Date']) - pd.to_datetime(from_100th_case['Min Date'])).dt.days



fig = px.line(from_100th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 100 case', height = 600)

fig.show()
gt_1000 = country_daywise[country_daywise['Confirmed']>1000]['Country'].unique()

temp = df[df['Country'].isin(gt_1000)]

temp = temp.groupby(['Country','Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>1000]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country','Min Date']



from_1000th_case = pd.merge(temp, min_date, on = 'Country')

from_1000th_case['N days'] = (pd.to_datetime(from_1000th_case['Date']) - pd.to_datetime(from_1000th_case['Min Date'])).dt.days



fig = px.line(from_1000th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 1000 case', height = 600)

fig.show()
gt_10000 = country_daywise[country_daywise['Confirmed']>10000]['Country'].unique()

temp = df[df['Country'].isin(gt_10000)]

temp = temp.groupby(['Country','Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>10000]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country','Min Date']



from_10000th_case = pd.merge(temp, min_date, on = 'Country')

from_10000th_case['N days'] = (pd.to_datetime(from_10000th_case['Date']) - pd.to_datetime(from_10000th_case['Min Date'])).dt.days



fig = px.line(from_10000th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 10000 case', height = 600)

fig.show()
gt_100000 = country_daywise[country_daywise['Confirmed']>100000]['Country'].unique()

temp = df[df['Country'].isin(gt_100000)]

temp = temp.groupby(['Country','Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>100000]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country','Min Date']



from_100000th_case = pd.merge(temp, min_date, on = 'Country')

from_100000th_case['N days'] = (pd.to_datetime(from_100000th_case['Date']) - pd.to_datetime(from_100000th_case['Min Date'])).dt.days



fig = px.line(from_100000th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 100000 case', height = 600)

fig.show()
gt_1000000 = country_daywise[country_daywise['Confirmed']>1000000]['Country'].unique()

temp = df[df['Country'].isin(gt_1000000)]

temp = temp.groupby(['Country','Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Confirmed']>1000000]



min_date = temp.groupby('Country')['Date'].min().reset_index()

min_date.columns = ['Country','Min Date']



from_1000000th_case = pd.merge(temp, min_date, on = 'Country')

from_1000000th_case['N days'] = (pd.to_datetime(from_1000000th_case['Date']) - pd.to_datetime(from_1000000th_case['Min Date'])).dt.days



fig = px.line(from_1000000th_case, x = 'N days', y = 'Confirmed', color = 'Country', title = 'N days from 1000000 case', height = 600)

fig.show()
full_latest = df[df['Date'] == max(df['Date'])]

fig = px.treemap(full_latest.sort_values(by = 'Confirmed', ascending = False).reset_index(drop = True),

                 path = ['Country'], values = 'Confirmed', height = 700,

                 title = 'Number of Confirmed Cases', color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()
full_latest = df[df['Date'] == max(df['Date'])]

fig = px.treemap(full_latest.sort_values(by = 'Deaths', ascending = False).reset_index(drop = True),

                 path = ['Country'], values = 'Deaths', height = 700,

                 title = 'Number of Deaths Reported', color_discrete_sequence = px.colors.qualitative.Dark2)

fig.data[0].textinfo = 'label+text+value'

fig.show()
first_date = df[df['Confirmed']>0]

first_date = first_date.groupby('Country')['Date'].agg(['min']).reset_index()



last_date = df.groupby(['Country','Date'])['Confirmed','Deaths','Recovered']

last_date = last_date.sum().diff().reset_index()



mask = (last_date['Country'] != last_date['Country'].shift(1))



last_date.loc[mask,'Confirmed'] = np.nan

last_date.loc[mask,'Deaths'] = np.nan

last_date.loc[mask,'Recovered'] = np.nan

last_date = last_date[last_date['Confirmed']>0]

last_date = last_date.groupby('Country')['Date'].agg(['max']).reset_index()



first_last = pd.concat([first_date, last_date['max']], axis = 1)

first_last['Days'] = pd.to_datetime(first_last['max']) - pd.to_datetime(first_last['min'])

first_last['Task'] = first_last['Country']

first_last.columns = ['Country','Start','Finish','Days','Task']

first_last = first_last.sort_values('Days')



colors = ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in range(len(first_last))]

fig = ff.create_gantt(first_last, index_col = 'Country', colors = colors, show_colorbar = False,

                      bar_width = 0.2, showgrid_x = True, showgrid_y = True, height = 2500)

fig.show()
temp = country_daywise.groupby(['Country','Date'])['Confirmed'].sum().reset_index()

temp = temp[temp['Country'].isin(gt_100000)]

countries = temp['Country'].unique()



ncols = 3

nrows = math.ceil(len(countries)/ncols)



fig = make_subplots(rows = nrows, cols = ncols, shared_xaxes = False, subplot_titles = countries)



for ind, country in enumerate(countries):

    row = int((ind/ncols)+1)

    col = int((ind%ncols)+1)

    fig.add_trace(go.Bar(x = temp['Date'], y = temp.loc[temp['Country']==country,'Confirmed'], name = country), 

                  row = row, col = col)

fig.update_layout(height = 4000, title_text = 'Confirmed Cases in Each Country')

fig.update_layout(showlegend = False)

fig.show()