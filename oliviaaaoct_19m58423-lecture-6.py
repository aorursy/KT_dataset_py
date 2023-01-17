import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold =np.inf)



Selected_country = 'South Korea'

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header =0)

#print (np.unique(df['Country/Region'].values))

df = df[df['Country/Region']==Selected_country]

#df3 = df.groupby('ObservationDate').sum()

#print(df3)

df['Daily_confirmed'] = (df['Confirmed'].diff())

df['Daily_deaths'] = (df['Deaths'].diff())

df['Daily_recovery'] = (df['Recovered'].diff())

#df['Daily_confirmed'] .plot()

#plt.show()
from plotly.offline import iplot

import plotly.graph_objs as go



Daily_confirmed_object = go.Scatter(x=df.ObservationDate, y=df['Daily_confirmed'].values,name='Daily confirmed')

Daily_deaths_object = go.Scatter(x=df.ObservationDate, y=df['Daily_deaths'].values,name='Daily deaths')

Daily_recoveries_object = go.Scatter(x=df.ObservationDate, y=df['Daily_recovery'].values,name='Daily recoveries')



layout_object = go.Layout(title = 'South Korea daily cases 19M58423',xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[Daily_confirmed_object,Daily_deaths_object,Daily_recoveries_object],layout=layout_object)

iplot(fig)
df1 = df.iloc[:,[1,8,9,10]]

df1 = df1.fillna(0.)

styled_object = df1.style.background_gradient(cmap='gist_ncar').highlight_max('Daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

#f = open('table_19M58423.html','w')

#f.write(styled_object.render())



dfx = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header =0)



rk = dfx[dfx['ObservationDate']=='06/06/2020']

rk = rk.groupby('Country/Region').sum()

rk = rk.sort_values(by='Confirmed',ascending=False).reset_index() 



print('Ranking of South Korea: ', rk[rk['Country/Region']=='South Korea'].index.values[0]+1)