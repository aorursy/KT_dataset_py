import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)



selected_country='Brazil'

df = pd.read_csv("../input/novel-corona-virus-2019-dataset/covid_19_data.csv",header=0)

df = df[df['Country/Region']==selected_country]

df = (df.groupby('ObservationDate').sum())



print(df)
df['daily_confirmed'] = df['Confirmed'].diff()

df['daily_deaths'] = df['Deaths'].diff()

df['daily_recovered'] = df['Recovered'].diff()

print(df)
df['daily_confirmed'].plot(color='blue')

df['daily_recovered'].plot(color='yellow')

df['daily_deaths'].plot(color='red')

plt.show()
from plotly.offline import iplot

import plotly.graph_objs as go



daily_confirmed_object = go.Scatter(x=df.index,y=df['daily_confirmed'].values,name='Daily Confirmed')

daily_deaths_object = go.Scatter(x=df.index,y=df['daily_deaths'].values,name='Daily Deaths')

daily_recovered_object = go.Scatter(x=df.index,y=df['daily_recovered'].values,name='Daily Recovered')



layout_object = go.Layout(title='Brazil Daily Case 19M58558',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recovered_object],layout=layout_object)

iplot(fig)

fig.write_html('Brazil_Daily_Case_19M58558.html')
df1 = df#['daily_confirmed']

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='Pastel1').highlight_max('daily_confirmed').set_caption('Daily_Summaries')

display(styled_object)

f = open('Table_19M58558.html','w')

f.write(styled_object.render())
df = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/covid_19_data.csv')

df.index=df['ObservationDate']

df = df.drop(['SNo','ObservationDate'],axis=1)

df_Brazil = df[df['Country/Region']=='Brazil']



latest = df[df.index=='06/16/2020']

latest = latest.groupby('Country/Region').sum()

latest = latest.sort_values(by='Confirmed',ascending=False).reset_index()



print('Brazil Rank: ', latest[latest['Country/Region']=='Brazil'].index.values[0]+1)