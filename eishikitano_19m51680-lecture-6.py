import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

np.set_printoptions(threshold=np.inf)
df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)#0 mean first low is header

#print(np.unique(df['Country/Region'].values))

df
selelcted_country='Chile'

Chile_data=df[df['Country/Region']==selelcted_country]

Chile_data=Chile_data.groupby('ObservationDate').sum()

print(Chile_data)
Chile_data['daily_confirmed'] = Chile_data['Confirmed'].diff()

Chile_data['daily_deaths']=Chile_data['Deaths'].diff()

Chile_data['daily_recovery']=Chile_data['Recovered'].diff()

Chile_data['pop_rate_confirmed']=Chile_data['Confirmed']/19116209*100000

Chile_data['pop_rate_deaths']=Chile_data['Deaths']/19116209*100000

Chile_data['Death_rate']=Chile_data['Deaths']/Chile_data['Confirmed']
print(Chile_data)
selelcted_country='Brazil'

Brazil_data=df[df['Country/Region']==selelcted_country]

Brazil_data=Brazil_data.groupby('ObservationDate').sum()

print(Brazil_data)
Brazil_data['daily_confirmed'] = Brazil_data['Confirmed'].diff()

Brazil_data['daily_deaths']=Brazil_data['Deaths'].diff()

Brazil_data['daily_recovery']=Brazil_data['Recovered'].diff()

Brazil_data['pop_rate_confirmed']=Brazil_data['Confirmed']/212559409*1000000

Brazil_data['pop_rate_deaths']=Brazil_data['Deaths']/212559409*100000

Brazil_data['Death_rate']=Brazil_data['Deaths']/Brazil_data['Confirmed']
print(Brazil_data)
from plotly.offline import iplot

import plotly.graph_objs as go



# Chile daily cases

daily_confirmed_object = go.Scatter(x=Chile_data.index, y=Chile_data['daily_confirmed'].values, name='Daily_confirmed')

daily_deaths_object = go.Scatter(x=Chile_data.index, y=Chile_data['daily_deaths'].values, name='Daily_deaths')



layout_object = go.Layout(title='Chile daily cases 19M51680', xaxis=dict(title='Dete'),yaxis=dict(title='Number of people'))

fig = go.Figure(data=[daily_confirmed_object,daily_deaths_object],layout=layout_object)

iplot(fig)

fig.write_html('Chile daily cases 19M51680.html')
# Compare Chile and Brazil

Chile_pop_rate_deaths_object = go.Scatter(x=Chile_data.index, y=Chile_data['pop_rate_deaths'].values, name='Chile')

Brazil_pop_rate_deaths_object = go.Scatter(x=Brazil_data.index, y=Brazil_data['pop_rate_deaths'].values, name='Brazil')



layout_object2 = go.Layout(title='Chile and Brazil (death_rate per 100,000 people) 19M51680', xaxis=dict(title='Dete'),yaxis=dict(title='Number of people'))

fig2 = go.Figure(data=[Brazil_pop_rate_deaths_object,Chile_pop_rate_deaths_object],layout=layout_object2)

iplot(fig2)

fig.write_html('Chile and Brazil(death_rate per100,000)19M51680.html')
# Make a table of chile's situation 

Chile_data2=Chile_data[['daily_confirmed','daily_deaths','daily_recovery','Recovered']]

Chile_data2=Chile_data2.fillna(0.)

styled_object = Chile_data2.style.background_gradient(cmap='gist_ncar').highlight_max('daily_confirmed').set_caption('Daily Summaries')

display(styled_object)

f=open('Chile_data_table_19M51680.html','w')

f.write(styled_object.render())
# Make Global ranking table

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)

df1= df[df['ObservationDate']=='06/12/2020']

df1_rank=df1.groupby(['Country/Region']).sum().sort_values(by=['Confirmed'],ascending=False).reset_index()

df1_rank=df1_rank[['Country/Region','Confirmed','Deaths','Recovered']]

df1_rank_object=df1_rank#.style

display(df1_rank_object)

#df1_rank_object.to_csv('ranking.csv')

df1_rank_object.to_html('Global_ranking_19M51680.html')
# Calculate death and confirmed rate per 100,000 people

df2 = pd.read_csv('../input/homework2/ranking.csv',header=0)

df2['pop_rate_confirmed']=df2['Confirmed']/df2['population']*100000

df2['pop_rate_deaths']=df2['Deaths']/df2['population']*100000

df2=df2.sort_values(by=['pop_rate_deaths'],ascending=False).reset_index()

df2.to_html('pop_rate_Global_ranking_19M51680.html')

df2
# Chile's rank

rank=df2.groupby(['Country/Region']).sum()

rank=rank.rank(ascending=False)

print(rank.loc['Chile'])