import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country='Philippines'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()

print(df)
df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovery'] = df['Recovered'].diff() 

df['daily_confirmed'].plot() #blue

df['daily_deaths'].plot() #orange

df['daily_recovery'].plot() #green

plt.show()
print(df)
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily deaths')

daily_recovery_object = go.Scatter(x=df.index,y=df['daily_recovery'].values,name='Daily recovery')



layout_object = go.Layout(title='Philippines daily cases 17B08669',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovery_object],layout=layout_object)

iplot(fig)

fig.write_html('Philippines_daily_cases_17B08669.html')
df1 = df

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='jet').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_17B08669.html','w')

f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df.index=df['ObservationDate']

latest = df[df.index=='06/10/2020']

latest = latest.groupby(['Country/Region']).sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()

print('Ranking of Philippines: ', latest[latest['Country/Region']=='Philippines'].index.values[0]+1)