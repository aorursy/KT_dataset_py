import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country ='Brazil'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df = df[df['Country/Region']==selected_country]

df = df.groupby('ObservationDate').sum()

print(df)



df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovery'] = df['Recovered'].diff()



df['daily_confirmed'].plot()

df['daily_deaths'].plot()

df['daily_recovery'].plot()

plt.show()
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index, y=df['daily_confirmed'].values, name = 'Daily Confirmed')

daily_deaths_object = go.Scatter(x=df.index, y=df['daily_deaths'].values, name = 'Daily Death')

daily_recovered_object = go.Scatter(x=df.index, y=df['daily_recovery'].values, name = 'Daily Recovery')



layout_object = go.Layout(title='Brazil Daily cases 20M51636', xaxis = dict(title='Date'), yaxis = dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object, daily_deaths_object, daily_recovered_object], layout = layout_object)

iplot(fig)

fig.write_html('Brazil_daily_cases_20M51636.html')
df1 = df#[['daily_confirmed']]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f = open('table_20M51636.html','w')

f.write(styled_object.render())
data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

data.index=data['ObservationDate']

data = data.drop(['SNo','ObservationDate'],axis=1)





latest = data[data.index=='06/10/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 



#print(latest)

print('Ranking of Brazil: ', latest[latest['Country/Region']=='Brazil'].index.values[0]+1)
