import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country='Mexico'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()

print(df)
df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovery'] = df['Recovered'].diff()

df['daily_confirmed'].plot()

df['daily_recovery'].plot()

df['daily_deaths'].plot()

plt.show()
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

daily_recovered_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recovered')



layout_object = go.Layout(title='Mexico daily cases 20M52067',xaxis=dict(title='Date'),yaxis=dict(title='Number of People'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)

iplot(fig)

fig.write_html('Mexico_daily_case_20M52067.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M52067.html','w')

f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df.index=df['ObservationDate']

df = df.drop(['SNo','ObservationDate'],axis=1)

df.head()

df_Mexico = df[df['Country/Region']=='Mexico']

df_Mexico.tail()

latest = df[df.index=='06/12/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 



print('Ranking of Mexico: ', latest[latest['Country/Region']=='Mexico'].index.values[0]+1)