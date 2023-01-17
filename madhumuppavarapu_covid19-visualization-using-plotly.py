from IPython.display import Image

Image('../input/covid19/covid.jpg')
# importing necessary modules for Data Visualization and Exploratory Data Analytics

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

%matplotlib inline

import plotly.express as px

from plotly.subplots import make_subplots

import plotly.graph_objects as go

import seaborn as sns

import warnings

warnings.filterwarnings("ignore")

# Overview of data

world_data=pd.read_csv('../input/corona-virus-report/worldometer_data.csv')

day_data=pd.read_csv('../input/corona-virus-report/day_wise.csv')

world_data.head()
world_data.isnull().sum()
# As most of the records in NewDeaths,NewCases,NewRecovered are null values so we'll drop all those columns

world_data.drop(columns=['NewCases','NewDeaths','NewRecovered'],inplace=True)

world_data['cases_per_tests']=world_data['TotalCases']/world_data['TotalTests']

world_data.head()
sns.pairplot(world_data,hue="Continent")

plt.show()
sns.pairplot(day_data)

plt.show()
plt.figure(figsize=(10,10))

sns.heatmap(world_data.isnull(),fmt='d',annot=True)
fig = px.treemap(world_data, path=["Country/Region"], values="TotalCases", height=700,

                 title="TotalCases of each country in a tree map")



fig.show()
fig = px.treemap(world_data, path=["Country/Region"], values="ActiveCases", height=700,

                 title="ActiveCases of each country in a tree map")



fig.show()
fig = px.treemap(world_data, path=["Country/Region"], values="TotalRecovered", height=700,

                 title="TotalRecovered of each country in a tree map representation")



fig.show()
fig = px.choropleth(world_data, locations="Country/Region", locationmode='country names', 

                  color="Country/Region", hover_name="Country/Region", 

                   hover_data=["Country/Region","TotalCases"],title='Total Cases data representation in a map')

fig.show()
fig = px.choropleth(world_data, locations="Country/Region", locationmode='country names', 

                  color="Country/Region", hover_name="Country/Region", title='Total Recovered cases ',

                   hover_data=["Country/Region","TotalRecovered"])

fig.show()
fig = px.choropleth(world_data, locations="Country/Region", locationmode='country names', 

                  color="Country/Region", hover_name="Country/Region", title='active cases',

                   hover_data=["Country/Region","ActiveCases"])

fig.show()
fig = px.choropleth(world_data, locations="Country/Region", locationmode='country names', 

                  color="Country/Region", hover_name="Country/Region", title='Total Deaths',

                   hover_data=["Country/Region","TotalDeaths"])

fig.show()
fig = px.choropleth(world_data, locations="Country/Region", locationmode='country names', 

                  color="Country/Region", hover_name="Country/Region", title='COVID-19 details in a map representation',

                   hover_data=["Country/Region","Continent","TotalCases","ActiveCases","TotalDeaths","TotalRecovered"])

fig.show()
fig = px.choropleth(world_data, locations="Country/Region", locationmode='country names', 

                  color="Country/Region", hover_name="Country/Region", 

                   hover_data=["TotalTests","Tests/1M pop","Deaths/1M pop","cases_per_tests"],title='map representation on testing of covid data')

fig.show()
fig = px.scatter_geo(world_data, locations="Country/Region",

                     size="TotalCases", # size of markers, "pop" is one of the columns of gapminder

                     )

fig.show()
fig=go.Figure(go.Scattergeo())

fig.update_geos(projection_type="orthographic",showrivers=True)

fig.update_layout(height=500)
day_data.head()
def pie_chart_rep(values,names,title):

    figure=plt.figure(figsize=(30,30))

    fig = px.pie(world_data, values=values, names=names, title=title)

    fig.update_layout()

    fig.show()
pie_chart_rep(values='TotalCases',names='Country/Region',title='TotalCases in each Country/Region')

pie_chart_rep(values=world_data['TotalCases'],names=world_data['Continent'],title='TotalCases in each Continent')
world_data['WHO Region'].fillna('WHO')
pie_chart_rep(values=world_data['TotalCases'],names=world_data['WHO Region'],title='TotalCases based on WHO Region')
fig = px.scatter(world_data, x="TotalDeaths", y="Serious,Critical",color='Country/Region',

               hover_name="Country/Region",title='line plot b/w TotalDeaths and Serious,Critical')

fig.show()
fig = go.Figure()

fig.add_trace(go.Scatter(x=world_data['Country/Region'], y=world_data['TotalCases'],

                    mode='lines',

                    name='TotalCases'))

fig.add_trace(go.Scatter(x=world_data['Country/Region'], y=world_data['ActiveCases'],

                    mode='lines+markers',

                    name='ActiveCases'))

fig.add_trace(go.Scatter(x=world_data['Country/Region'], y=world_data['TotalRecovered'],

                    mode='markers', name='TotalRecovered'))

fig.update_layout(title='Country vs cases(Total,Active,Recovered)')

fig.show()
fig=px.bar(y=world_data['Country/Region'].iloc[0:25],x=world_data['TotalCases'].iloc[:25],title='Country/Region Vs TotalCases',color=world_data['Country/Region'].iloc[:25])

fig.show()
fig=px.bar(y=world_data['Country/Region'].iloc[:25],x=world_data['ActiveCases'].iloc[:25],title='Country/Region Vs ActiveCases',color=world_data['Country/Region'].iloc[:25])

fig.show()
fig=px.bar(y=world_data['Country/Region'].iloc[:25],x=world_data['TotalRecovered'].iloc[:25],title='Country/Region Vs TotalRecovered',color=world_data['Country/Region'].iloc[:25])

fig.show()
fig=px.scatter(y=world_data['Population'],x=world_data['TotalCases'],title='Popilation Vs TotalCases',color=world_data['Country/Region'])

fig.show()
sns.jointplot(y=world_data['Population'],x=world_data['Deaths/1M pop'],kind='reg')
sns.jointplot(data=day_data,y='Recovered',x='Recovered / 100 Cases',kind='kde')
day_data.head()
fig=px.scatter(y=world_data['Tot Cases/1M pop'],x=world_data['Deaths/1M pop'],title='Popilation Vs TotalCases',color=world_data['Country/Region'])

fig.show()
sns.jointplot(y=world_data['Tot Cases/1M pop'],x=world_data['Deaths/1M pop'],kind='reg')
fig=px.bar(world_data,y=world_data['Country/Region'].iloc[:25],x=[world_data['TotalCases'].iloc[:25],world_data['ActiveCases'].iloc[:25],world_data['TotalRecovered'].iloc[:25]],title='Country/Region Vs TotalRecovered',height=1000,orientation='h')

fig.show()
fig=px.bar(world_data,y=world_data['Country/Region'].iloc[:25],x=world_data['TotalTests'].iloc[:25],orientation='h',color=world_data['Country/Region'].iloc[:25],title='total covid tests conducted by top 25 countries')

fig.show()
fig=px.bar(world_data,y=world_data['Country/Region'].iloc[:25],x=world_data['cases_per_tests'].iloc[:25],orientation='h',color=world_data['Country/Region'].iloc[:25],title='total covid tests conducted by top 25 countries')

fig.show()
px.bar(world_data,y=world_data['Country/Region'],x=world_data['cases_per_tests'],color=world_data['Country/Region'],orientation='h',height=1500,title='Ratio of TotalCases to the TotalTestsConducted')
fig = go.Figure()

fig.add_trace(go.Scatter(x=world_data.ActiveCases, y=world_data.TotalCases,

                    mode='markers',

                    name='markers',text=world_data['Country/Region']))

fig.add_trace(go.Scatter(x=world_data.TotalRecovered, y=world_data.TotalCases,

                    mode='markers',

                    name='markers',text=world_data['Country/Region']))

fig.add_trace(go.Scatter(x=world_data.TotalDeaths, y=world_data.TotalCases,

                    mode='markers',

                    name='markers',text=world_data['Country/Region']))



fig.show()

fig=px.bar(day_data,x="Date",y="Confirmed")

fig.show()
def plot_of_each_day(y,title):

    fig=px.bar(day_data,x="Date",y=y,title=title,color="Date")

    fig.update_layout()

    fig.show()
plot_of_each_day(y="Confirmed",title='Confirmed cases on each day')
plot_of_each_day(y="Deaths",title='Deaths on each day')
plot_of_each_day(y="Recovered",title='Recovered cases on each day')
plot_of_each_day(y="Active",title='Active cases on each day')
def line_plot_of_each_day(y,title):

    fig=px.line(day_data,x="Date",y=y,title=title)

    fig.show()
line_plot_of_each_day(y="Deaths / 100 Cases",title='Deaths / 100 Cases on each day')
line_plot_of_each_day(y="Recovered / 100 Cases",title='Recovered / 100 Cases on each day')
line_plot_of_each_day(y="Deaths / 100 Recovered",title='Deaths / 100 Recovered on each day')
country_data=pd.read_csv('../input/corona-virus-report/country_wise_latest.csv')

fig=plt.figure(figsize=(5,5))

country_data["Confirmed"].plot(kind='kde',subplots=True,ax=fig.gca())

plt.legend()

plt.show()
def pie_chart_of_each_country(values,names,title):

    figure=plt.figure(figsize=(30,30))

    fig = px.pie(country_data, values=values, names=names,color="Country/Region", title=title)

    fig.update_layout()

    fig.show()
pie_chart_of_each_country(values="Confirmed last week",names="Country/Region",title='Confirmed last week of each country')
pie_chart_of_each_country(values="1 week change",names="Country/Region",title='1 week change of each country')
fig=px.line(country_data,x="Country/Region",y="New cases",color="WHO Region")

fig.show()
fig=px.line(country_data,x="Country/Region",y="New deaths",color="WHO Region")

fig.show()
fig=px.line(country_data,x="Country/Region",y="New recovered",color="WHO Region")

fig.show()
fig = go.Figure(data=go.Scatter(

    x=world_data['Population'],

    y=world_data['TotalCases'],

    mode='markers',

    marker=dict(size=[40, 60, 80, 100],

                color=[0, 1, 2, 3])

))



fig.show()
fig = go.Figure(data=go.Scatter(

    x=world_data['TotalCases'],

    y=world_data['TotalDeaths'],

    mode='markers',

    marker=dict(size=[40, 60, 80, 100],

                color=[0, 1, 2, 3])

))



fig.show()
fig = go.Figure(data=go.Scatter(

    x=world_data['TotalCases'],

    y=world_data['ActiveCases'],

    mode='markers',

    marker=dict(size=[40, 60, 80, 100],

                color=[0, 1, 2, 3])

))



fig.show()
px.scatter(world_data,x='Country/Region',y='TotalCases',color='Country/Region',hover_data=['Country/Region', 'Continent', 'Population', 'TotalCases',

       'TotalDeaths', 'TotalRecovered', 'ActiveCases', 'Serious,Critical',

       'Tot Cases/1M pop', 'Deaths/1M pop', 'TotalTests', 'Tests/1M pop',

       'WHO Region', 'cases_per_tests'])
fig=plt.figure(figsize=(20,20))

world_data.plot(kind='kde',subplots=True,ax=fig.gca())

plt.show()
fig=plt.figure(figsize=(20,20))

day_data.plot(kind='kde',subplots=True,ax=fig.gca())

plt.show()