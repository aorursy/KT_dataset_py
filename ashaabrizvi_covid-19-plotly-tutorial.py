import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import plotly

import plotly.express as px

import plotly.graph_objects as go
data = pd.read_csv('../input/corona-virus-report/full_grouped.csv')

data.head()
worldometer = pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

worldometer.head()
# We will use this data in Choropleth Map

latest_data = pd.read_csv('https://raw.githubusercontent.com/datasets/covid-19/master/data/countries-aggregated.csv')

latest_data.head()
# Making Another Dataframe for India

# We will use this in Line Plot and Scatter Plot



data_INDIA = data[data['Country/Region']=='India']

data_INDIA = data_INDIA[data_INDIA.Confirmed > 0]

data_INDIA.head()
#Data Table

worldometer_new=worldometer.drop(['NewCases','NewDeaths','NewRecovered'],axis=1)

worldometer_new = worldometer_new[worldometer_new['TotalCases'] > 10000]

worldometer_new = worldometer_new.groupby('Country/Region').sum()

worldometer_new.style.background_gradient(cmap='RdPu')
# Choropleth Map of the World

fig = px.choropleth(latest_data,locations='Country',locationmode='country names',color='Confirmed',animation_frame='Date')

fig.update_layout(title='Choropleth Map of Confirmed Cases - World on 13-10-2020',template="plotly_dark")

fig.show()
# Continent Map using Choropleth

fig = px.choropleth(latest_data,locations='Country',locationmode='country names',color='Confirmed',animation_frame='Date',scope='asia')

fig.update_layout(title='Choropleth Map of Confirmed Cases - Asia on 13-10-2020',template="plotly_dark")

fig.show()
# Continent Map using Choropleth

fig = px.choropleth(latest_data,locations='Country',locationmode='country names',color='Confirmed',animation_frame='Date',scope='europe')

fig.update_layout(title='Choropleth Map of Confirmed Cases - Europe on 13-10-2020',template="plotly_dark")

fig.show()
India_lockdown_1 = '2020-03-25'

India_lockdown_2 = '2020-04-15'

India_lockdown_3 = '2020-05-04'

India_lockdown_4 = '2020-05-18'
fig = px.line(data_INDIA,x='Date',y = 'New cases',title='Lockdown Phases - Infection Rate')

fig.add_shape(dict(type='line',

                   x0=India_lockdown_1,

                   y0=0,

                   x1=India_lockdown_1,

                   y1=data_INDIA['New cases'].max(),

                   line = dict(color='red',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_1,

                   y = data_INDIA['New cases'].max(),

                   text = 'Phase 1')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_2,

                   y0=0,

                   x1=India_lockdown_2,

                   y1=data_INDIA['New cases'].max(),

                   line = dict(color='blue',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_2,

                   y = data_INDIA['New cases'].max(),

                   text = 'Phase 2')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_3,

                   y0=0,

                   x1=India_lockdown_3,

                   y1=data_INDIA['New cases'].max(),

                   line = dict(color='yellow',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_3,

                   y = data_INDIA['New cases'].max(),

                   text = 'Phase 3')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_4,

                   y0=0,

                   x1=India_lockdown_4,

                   y1=data_INDIA['New cases'].max(),

                   line = dict(color='green',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_4,

                   y = data_INDIA['New cases'].max(),

                   text = 'Phase 4')

)
fig = px.line(data_INDIA,x='Date',y = 'New deaths',title='Lockdown Phases - Death Rate')

fig.add_shape(dict(type='line',

                   x0=India_lockdown_1,

                   y0=0,

                   x1=India_lockdown_1,

                   y1=data_INDIA['New deaths'].max(),

                   line = dict(color='red',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_1,

                   y = data_INDIA['New deaths'].max(),

                   text = 'Phase 1')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_2,

                   y0=0,

                   x1=India_lockdown_2,

                   y1=data_INDIA['New deaths'].max(),

                   line = dict(color='blue',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_2,

                   y = data_INDIA['New deaths'].max(),

                   text = 'Phase 2')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_3,

                   y0=0,

                   x1=India_lockdown_3,

                   y1=data_INDIA['New deaths'].max(),

                   line = dict(color='yellow',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_3,

                   y = data_INDIA['New deaths'].max(),

                   text = 'Phase 3')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_4,

                   y0=0,

                   x1=India_lockdown_4,

                   y1=data_INDIA['New deaths'].max(),

                   line = dict(color='green',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_4,

                   y = data_INDIA['New deaths'].max(),

                   text = 'Phase 4')

)

fig = px.line(data_INDIA,x='Date',y = 'New recovered',title='Lockdown Phases - Recovery Rate')

fig.add_shape(dict(type='line',

                   x0=India_lockdown_1,

                   y0=0,

                   x1=India_lockdown_1,

                   y1=data_INDIA['New recovered'].max(),

                   line = dict(color='red',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_1,

                   y = data_INDIA['New recovered'].max(),

                   text = 'Phase 1')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_2,

                   y0=0,

                   x1=India_lockdown_2,

                   y1=data_INDIA['New recovered'].max(),

                   line = dict(color='blue',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_2,

                   y = data_INDIA['New recovered'].max(),

                   text = 'Phase 2')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_3,

                   y0=0,

                   x1=India_lockdown_3,

                   y1=data_INDIA['New recovered'].max(),

                   line = dict(color='yellow',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_3,

                   y = data_INDIA['New recovered'].max(),

                   text = 'Phase 3')

)

fig.add_shape(dict(type='line',

                   x0=India_lockdown_4,

                   y0=0,

                   x1=India_lockdown_4,

                   y1=data_INDIA['New recovered'].max(),

                   line = dict(color='green',dash="dashdot")

                   ))

fig.add_annotation(dict(

                   x = India_lockdown_4,

                   y = data_INDIA['New recovered'].max(),

                   text = 'Phase 4')

)

# Scatter Plot in Plotly

fig = go.Figure(data=go.Scatter(x=data_INDIA['Confirmed'],y=data_INDIA['Active'],mode='markers',marker=dict(size=10,color=data_INDIA['New cases'],showscale=True),text=data_INDIA['Country/Region']))

fig.update_layout(title='Scatter Plot for Confirmed v Active Cases',xaxis_title='Confirmed',yaxis_title='Active')



fig.show()
# Scatter Plot in Plotly

fig = go.Figure(data=go.Scatter(x=data_INDIA['Deaths'],y=data_INDIA['Recovered'],mode='markers',marker=dict(size=10,color=data_INDIA['New cases'],showscale=True),text=data_INDIA['Country/Region']))

fig.update_layout(title='Scatter Plot for Deaths v Recovery',xaxis_title='Deaths',yaxis_title='Recovery')



fig.show()
# Pie Chart in PLotly

fig = px.pie(worldometer.head(20),values='TotalDeaths',names='Country/Region',title='Percentage of Total Deaths in 20 Most Affected Countries')

fig.show()
# Using Text in Pie Chart

fig = px.pie(worldometer.head(20),values='TotalRecovered',names='Country/Region',title='Percentage of Total Recovered in 20 Most Affected Countries')

fig.update_traces(textposition='inside', textinfo='percent+label')

fig.show()
# Bar Charts

fig = px.bar(worldometer.head(10), y='TotalTests',x='Country/Region',color='WHO Region',height=400)

fig.update_layout(title='Comparison of Total Tests of 10 Most Affected Countries',xaxis_title='Country',yaxis_title='Total Tests',template="plotly_dark")

fig.show()
# Bar Charts

fig = px.bar(worldometer.head(10), y='Deaths/1M pop',x='Country/Region',color='WHO Region',height=400)

fig.update_layout(title='Comparison of Deaths/Million of 10 Most Affected Countries',xaxis_title='Country',yaxis_title='Deaths/Million',template="plotly_dark")

fig.show()
# Bar Charts

fig = px.bar(worldometer.head(10), y='Tot Cases/1M pop',x='Country/Region',color='WHO Region',height=400)

fig.update_layout(title='Comparison of Cases/Million of 10 Most Affected Countries',xaxis_title='Country',yaxis_title='Cases/Million',template="plotly_dark")

fig.show()
# Bar Charts

fig = px.bar(worldometer.head(10), y='Tests/1M pop',x='Country/Region',color='WHO Region',height=400)

fig.update_layout(title='Comparison of Tests/Million of 10 Most Affected Countries',xaxis_title='Country',yaxis_title='Tests/Million',template="plotly_dark")

fig.show()
# Bubble Chart using Plotly

fig = px.scatter(worldometer.head(50), x="TotalCases", y="TotalDeaths",size='Population',

	               color="Continent",

                 hover_name="Country/Region", log_x=True, size_max=60)

fig.update_layout(title='Bubble Plot for Total Cases v Total Deaths of 50 Most Affected Countries',xaxis_title='Cases',yaxis_title='Deaths')

fig.show()
# Sunburst Chart using Plotly

fig = px.sunburst(worldometer.head(50),path=['Continent','Country/Region','WHO Region'],values='Population',

                  color='ActiveCases',

                  color_continuous_scale='RdBu',

                  color_continuous_midpoint=np.average(worldometer.head(50)['Serious,Critical'], weights=worldometer.head(50)['Population']))

fig.update_layout(title='Sunburst Chart')

fig.show()
# 3D Plots

fig = px.scatter_3d(worldometer.head(20), x='TotalCases', y='TotalDeaths', z='TotalRecovered',

              color='Country/Region')

fig.update_layout(title='3D Plot of Total Cases, Total Deaths and Total Recovered of Top 20 Affected Countries')

fig.show()
from wordcloud import WordCloud
plt.subplots(figsize=(20,8))

wordcloud = WordCloud(background_color='black',width=1920,height=1080).generate(" ".join(worldometer.head(25)['Country/Region']))

plt.imshow(wordcloud)

plt.axis('off')

plt.savefig('cast.png')

plt.show()