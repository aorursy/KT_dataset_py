import pandas as pd



import numpy as np



import plotly.express as px
df=pd.read_csv('../input/covid-19/data.csv')



data=df.drop([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30])



data
latest = data[data['Date'] == max(data['Date'])]



latest = latest.groupby('Date')['Daily Confirmed'].max().reset_index()



fig = px.bar(data.sort_values('Daily Confirmed', ascending=False)[:172][::-1], 

             y='Daily Confirmed', x='Date', color="Daily Confirmed",

             title='Daily Confirmed Cases in India', text='Daily Confirmed')



fig.show()

latest = data[data['Date'] == max(data['Date'])]



latest = latest.groupby('Date')['Total Confirmed'].max().reset_index()



fig = px.bar(data.sort_values('Total Confirmed', ascending=False)[:172][::-1], 

             y='Total Confirmed', x='Date', color="Total Confirmed",

             title='Total Confirmed Cases in India', text='Total Confirmed')



fig.show()
latest = data[data['Date'] == max(data['Date'])]



latest = latest.groupby('Date')['Daily Recovered'].max().reset_index()



fig = px.bar(data.sort_values('Daily Recovered', ascending=False)[:172][::-1], 

             y='Daily Recovered', x='Date', color="Daily Recovered",

             title='Daily Recovered Cases in India', text='Daily Recovered')



fig.show()
latest = data[data['Date'] == max(data['Date'])]



latest = latest.groupby('Date')['Total Recovered'].max().reset_index()



fig = px.bar(data.sort_values('Total Recovered', ascending=False)[:172][::-1], 

             y='Total Recovered', x='Date', color="Total Recovered",

             title='Total Recovered Cases in India', text='Total Recovered')



fig.show()
latest = data[data['Date'] == max(data['Date'])]



latest = latest.groupby('Date')['Daily Deceased'].max().reset_index()



fig = px.bar(data.sort_values('Daily Deceased', ascending=False)[:172][::-1], 

             y='Daily Deceased', x='Date', color="Daily Deceased",

             title='Daily Deceased Members in India', text='Daily Deceased')



fig.show()
latest = data[data['Date'] == max(data['Date'])]



latest = latest.groupby('Date')['Total Deceased'].max().reset_index()



fig = px.bar(data.sort_values('Total Deceased', ascending=False)[:172][::-1], 

             y='Total Deceased', x='Date', color="Total Deceased",

             title='Total Deceased Members in India', text='Total Deceased')



fig.show()
data2=pd.read_csv('../input/covid-19/corona_data.csv')



data2
latest = data2[data2['Name of State'] == max(data2['Name of State'])]



latest = latest.groupby('Name of State')['Population'].max().reset_index()



fig = px.bar(data2.sort_values('Population', ascending=False)[:34][::-1], 

             x='Population', y='Name of State', color="Population",

             title='State Wise Population of India', text='Population', orientation='h',height=700)



fig.show()

latest = data2[data2['Name of State'] == max(data2['Name of State'])]



latest = latest.groupby('Name of State')['Active Cases'].max().reset_index()



fig = px.bar(data2.sort_values('Active Cases', ascending=False)[:34][::-1], 

             x='Active Cases', y='Name of State', color="Active Cases",

             title='State Wise PActive Cases in India', text='Active Cases', orientation='h',height=700)



fig.show()
latest = data2[data2['Name of State'] == max(data2['Name of State'])]



latest = latest.groupby('Name of State')['Recovered'].max().reset_index()



fig = px.bar(data2.sort_values('Recovered', ascending=False)[:34][::-1], 

             x='Recovered', y='Name of State', color="Recovered",

             title='State Wise Recovered Cases in India', text='Recovered', orientation='h',height=700)



fig.show()
latest = data2[data2['Name of State'] == max(data2['Name of State'])]



latest = latest.groupby('Name of State')['Deaths'].max().reset_index()



fig = px.bar(data2.sort_values('Deaths', ascending=False)[:34][::-1], 

             x='Deaths', y='Name of State', color="Deaths",

             title='State Wise Deaths in India', text='Deaths', orientation='h',height=700)



fig.show()
latest = data2[data2['Name of State'] == max(data2['Name of State'])]



latest = latest.groupby('Name of State')['Total Confirmed cases'].max().reset_index()



fig = px.bar(data2.sort_values('Total Confirmed cases', ascending=False)[:34][::-1], 

             x='Total Confirmed cases', y='Name of State', color="Total Confirmed cases",

             title='State Wise Total Confirmed cases in India', text='Total Confirmed cases', orientation='h',height=700)



fig.show()
latest = data2[data2['Name of State'] == max(data2['Name of State'])]



latest = latest.groupby('Name of State')['Recovered Rate'].max().reset_index()



fig = px.bar(data2.sort_values('Recovered Rate', ascending=False)[:34][::-1], 

             x='Recovered Rate', y='Name of State', color="Recovered Rate",

             title='State Wise Recovered Rate in India', text='Recovered Rate', orientation='h',height=700)



fig.show()
latest = data2[data2['Name of State'] == max(data2['Name of State'])]



latest = latest.groupby('Name of State')['Death Rate'].max().reset_index()



fig = px.bar(data2.sort_values('Death Rate', ascending=False)[:34][::-1], 

             x='Death Rate', y='Name of State', color="Death Rate",

             title='State Wise Death Rate in India', text='Death Rate', orientation='h',height=700)



fig.show()