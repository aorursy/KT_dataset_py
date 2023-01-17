import numpy as np

import matplotlib.pyplot as plt

import pandas as pd



selected_country = 'South Korea'



df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", header = 0)

#print(np.unique(df['Country/Region'].values))

df_korea = df[df['Country/Region'] == selected_country]

df_korea = df_korea.groupby('ObservationDate').sum()
df_korea['daily_confirmed'] = df_korea['Confirmed'].diff()

df_korea['daily_deaths'] = df_korea['Deaths'].diff()

df_korea['daily_recovered'] = df_korea['Recovered'].diff()
from plotly.offline import iplot,init_notebook_mode

import plotly.graph_objs as go

import plotly as py



init_notebook_mode(connected=True) 



daily_confirmed_object = go.Scatter(x = df_korea.index, y = df_korea['daily_confirmed'].values, name = 'Daily confirmed')

daily_deaths_object = go.Scatter(x = df_korea.index, y = df_korea['daily_deaths'].values, name = 'Daily deaths')

daily_recovered_object = go.Scatter(x = df_korea.index, y = df_korea['daily_recovered'].values, name = 'Daily recovered')



layout = go.Layout(title = "Covid-19 Cases in Sorth Korea", xaxis = dict(title = 'Date'), yaxis = dict(title = 'Case Number') )

figure = go.Figure(data = [daily_confirmed_object, daily_deaths_object, daily_recovered_object], layout = layout)

py.offline.iplot(figure)



figure.write_html("SouthKorea_daily_cases_figure_19M58417.html")
df_korea_copy = df_korea

df_korea_copy = df_korea_copy.fillna(0.)

styled_object = df_korea_copy.style.background_gradient(cmap = 'gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

file = open('SouthKorea_daily_cases_table_19M58417.html', 'w')

file.write(styled_object.render())
df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv", header = 0)

df_country = df[df['ObservationDate'] == '06/06/2020']

df_country = df_country.groupby(['Country/Region']).sum().sort_values(by='Confirmed',ascending=False).reset_index()

print(df_country[df_country['Country/Region'] == 'South Korea'].index.values[0] + 1)