import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df.index=df['ObservationDate']

df = df.drop(['SNo','ObservationDate'],axis=1)

#df.tail()
selected_country = 'Israel'
df_selected = df[df['Country/Region']==selected_country]

df_dates = df_selected.groupby('ObservationDate').sum()



#Calcurate daily numbers

df_dates['Daily_confirmed'] = df_dates['Confirmed'].diff()

df_dates['Daily_deaths'] = df_dates['Deaths'].diff()

df_dates['Daily_recovered'] = df_dates['Recovered'].diff()





#Plot time-series

from plotly.offline import iplot

import plotly.graph_objs as go



#Daily number

daily_confirmed_object = go.Scatter(x=df_dates.index, y=df_dates['Daily_confirmed'].values, name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df_dates.index, y=df_dates['Daily_deaths'].values,name='Daily deaths')

daily_recoveries_object = go.Scatter(x=df_dates.index, y=df_dates['Daily_recovered'].values,name='Daily recovered')



#Cumulative number

confirmed_object = go.Bar(x=df_dates.index, y=df_dates['Confirmed'].values, name='Cumulative confirmed',opacity=0.4)

deaths_object = go.Bar(x=df_dates.index, y=df_dates['Deaths'].values,name='Cumulative deaths',opacity=0.4)



layout_object = go.Layout(title=dict(text=f'Fig.1 {selected_country} daily cases (20M51659)',xref='paper',x=0.5,xanchor = 'center'),

                          xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recoveries_object,deaths_object],layout=layout_object)

iplot(fig)
#Colored tables

df1 = df_dates.fillna(0.).iloc[:,3:6]

styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('Daily_confirmed').set_caption('Table1: Daily Summaries')

styled_object
latest = df[df.index==f'{df.tail(1).index.values[0]}']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 



#Selected country's Ranking

print(f'Current ranking of {selected_country}: ', latest[latest['Country/Region']==f'{selected_country}'].index.values[0]+1, f' (As of {df.tail(1).index.values[0]})')
Dailycases = df_dates.sort_values(by='Daily_confirmed',ascending=False)



#Largest daily cases date

print(f'Lagest daily cases in {selected_country}: {Dailycases.head(1).index.values[0]}', f' (Number of confirmed people: {Dailycases.head(1)["Daily_confirmed"].values[0]})' )
df_dates['Daily_difference'] = df_dates['Daily_confirmed'].diff()

df2 = df_dates.fillna(0)

df2 = df2.loc[:, ["Daily_confirmed","Daily_difference"]]

styled_object1 = df2.style.bar(align='zero', color=['blue','red']).set_caption('Table2: Daily confirmed and difference')

display(styled_object1)