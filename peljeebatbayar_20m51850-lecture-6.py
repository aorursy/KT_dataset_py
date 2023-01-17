import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
np.set_printoptions(threshold=np.inf)

selected_country='Austria'
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
#print(np.unique(df['Country/Region'].values))
df = df[df['Country/Region']==selected_country]
df = df.groupby('ObservationDate').sum()
print(df)
#print(df['Confirmed'].diff())
#print(df['Deaths'].diff())
df['daily_confirmed']=df['Confirmed'].diff()
df['daily_deaths']=df['Deaths'].diff()
df['daily_recovery']=df['Recovered'].diff()
df['daily_confirmed'].plot()
df['daily_recovery'].plot()
plt.show()
print(df)
from plotly.offline import iplot
import plotly.graph_objs as go

daily_confirmed_object = go.Scatter(x=df.index, y=df['daily_confirmed'].values, name='Daily confirmed')
daily_deaths_object = go.Scatter(x=df.index, y=df['daily_deaths'].values, name='Daily deaths')

layout_object = go.Layout(title='Austria daily cases 20M51850', xaxis=dict(title='Date'), yaxis=dict(title='Number of people'))
fig = go.Figure(data=[daily_confirmed_object, daily_deaths_object], layout=layout_object)
iplot(fig)
fig.write_html('Austria_daily_cases_20M51850.html')
df1=df#[['daily_confirmed']]
df1=df1.fillna(0.)
styled_object = df1.style.background_gradient(cmap='gist_heat').highlight_max('daily_confirmed').set_caption('Daily Summaries')
display(styled_object)
f = open('table_20M51850.html','w')
f.write(styled_object.render())
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)
df1 = df.groupby(['ObservationDate','Country/Region']).sum()
df2 = df[df['ObservationDate']=='06/16/2020'].sort_values(by=['Confirmed'],ascending=False).reset_index()
print(df2[df2['Country/Region']=='Austria'])
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header=0)

df2 = df[df['Country/Region']=='Austria']
df2.tail()

df2 = df[df.index=='06/16/2020']
df2 = df2.groupby('Country/Region').sum()
df2 = df2.sort_values(by='Confirmed',ascending=False).reset_index() 

print('Ranking of Austria: ', df[df['Country/Region']=='Austria'].index.values[0]+1)

#I am not sure how to find global ranking of Austria. The result seems to be mistaken. 