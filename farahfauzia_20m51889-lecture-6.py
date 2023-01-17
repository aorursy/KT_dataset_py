import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
np.set_printoptions(threshold=np.inf)

#loading csv as data frame
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
df.head()
#print(df.columns)
#print(np.unique(df['Country/Region'].values)) #display unique values
dfT = df[df['Country/Region']=='Turkey']
dfT = dfT.groupby('ObservationDate').sum()
dfT.tail()
dfT['daily_confirmed'] = dfT['Confirmed'].diff() 
dfT['daily_deaths'] = dfT['Deaths'].diff()
dfT['daily_recovered'] = dfT['Recovered'].diff()
dfT.tail()

#diff substract from the row before for specific columns (obtain daily)
dfT['daily_confirmed'].plot()
dfT['daily_deaths'].plot()
dfT['daily_recovered'].plot()
plt.show()
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=dfT.index,y=dfT['daily_confirmed'].values,name='Daily Confirmed') #index is observed date
daily_deaths_object = go.Scatter(x=dfT.index,y=dfT['daily_deaths'].values,name='Daily Deaths')
daily_recoveries_object = go.Scatter(x=dfT.index,y=dfT['daily_recovered'].values,name='Daily Recoveries')

layout_object = go.Layout(title='Turkey Daily cases 20M51889', xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object,daily_recoveries_object],layout=layout_object)
iplot(fig)
fig.write_html('Turkey_daily_cases_20M51889.html')
dfT2 = dfT[['daily_confirmed']]
dfT2 = dfT.fillna(0.)

styled_object = dfT2.style.background_gradient(cmap='nipy_spectral').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)

f = open('table_20M51889.html','w')
f.write(styled_object.render())
latest = df[df.ObservationDate=='06/06/2020']
latest = latest.groupby('Country/Region').sum()
latest = latest.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Global Ranking of Turkey: ', latest[latest['Country/Region']=='Turkey'].index.values[0]+1)

##script is modified from Ryza Rynazal's shares on Slack group##
print('Global Ranking of Italy: ', latest[latest['Country/Region']=='Italy'].index.values[0]+1)
#Obtaining daily confirmed & death cases for Italy
dfI = df[df['Country/Region']=='Italy']
dfI = dfI.groupby('ObservationDate').sum()
dfI['daily_confirmed'] = dfI['Confirmed'].diff() 
dfI['daily_deaths'] = dfI['Deaths'].diff()
dfI = dfI.fillna(0.)
dfI.tail()
#Comparing daily confirmed cases in Turkey & Italy
daily_confirmed_objectT = go.Scatter(x=dfT.index,y=dfT['daily_confirmed'].values,name='Daily Confirmed Turkey') 
daily_confirmed_objectI = go.Scatter(x=dfI.index,y=dfI['daily_confirmed'].values,name='Daily Confirmed Italy')

layout_object = go.Layout(title='Daily Confirmed Cases of COVID-19 in Turkey & Italy - 20M51889', xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_objectI,daily_confirmed_objectT],layout=layout_object)
iplot(fig)
fig.write_html('Turkey-Italy_daily_confirmed_cases_20M51889.html')

#Comparing daily death cases in Turkey & Italy
daily_deaths_objectT = go.Scatter(x=dfT.index,y=dfT['daily_deaths'].values,name='Daily Deaths Turkey')
daily_deaths_objectI = go.Scatter(x=dfI.index,y=dfI['daily_deaths'].values,name='Daily Deaths Italy')

layout_object = go.Layout(title='Daily Deaths of COVID-19 in Turkey & Italy - 20M51889', xaxis=dict(title='Date'),yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_deaths_objectI,daily_deaths_objectT],layout=layout_object)
iplot(fig)
fig.write_html('Turkey-Italy_daily_death_cases_20M51889.html')
