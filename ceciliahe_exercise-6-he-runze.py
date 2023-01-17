import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country='Canada'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df[df['Country/Region']=='Canada']

df = df.groupby('ObservationDate').sum()

print(df)
df['daily_confirmed_cases'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recoveries'] = df['Recovered'].diff()

df['daily_confirmed_cases'].plot()

df['daily_recoveries'].plot()

df['daily_deaths'].plot()

plt.show()

print(df)

from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_cases_object = go.Scatter(x=df.index,y=df['daily_confirmed_cases'].values,name='Daily confirmed cases')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

daily_recoveries_object = go.Scatter(x=df.index,y=df['daily_recoveries'].values,name='Daily recoveries')



layout_object = go.Layout(title='Canada daily cases 20M51903',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_cases_object,daily_deaths_object,daily_recoveries_object],layout=layout_object)

iplot(fig)





fig.write_html('Canada daily cases 20M51903.html')
df1 = df#[['daily_confirmed_cases']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='RdPu').highlight_max('daily_confirmed_cases').highlight_max('daily_recoveries').highlight_max('daily_deaths').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51903.html','w')

f.write(styled_object.render())

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df.index=df['ObservationDate']

df = df.drop(['SNo','ObservationDate'],axis=1)

df.head()



latest = df[df.index=='06/10/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 



print('Ranking of Canada: ', latest[latest['Country/Region']=='Canada'].index.values[0]+1)
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df.index=df['ObservationDate']

df = df.drop(['SNo','ObservationDate'],axis=1)

df.head()



latest = df[df.index=='06/10/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Deaths',ascending=False).reset_index() 



print('Ranking of Canada: ', latest[latest['Country/Region']=='Canada'].index.values[0]+1)