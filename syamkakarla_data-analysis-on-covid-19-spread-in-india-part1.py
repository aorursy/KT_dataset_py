# Standard plotly imports

import plotly as py

import plotly.express as px

import plotly.graph_objs as go

import plotly.figure_factory as ff

from plotly.offline import iplot, init_notebook_mode

# Using plotly + cufflinks in offline mode

import cufflinks

cufflinks.go_offline(connected=True)

init_notebook_mode(connected=True)

import pandas as pd

import numpy as np



path = '../input/covid19-in-india/'
w = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')

x = w.groupby('Country/Region')['Confirmed'].max()



w_x = pd.DataFrame(data={'Country':x.index, 'Confirmed':x.values})

# w_x.head()

data = dict(type='choropleth',

            locations = w_x.Country.values.tolist(),

            locationmode = 'country names',

            colorscale = 'sunsetdark',

            text = w_x.Country.values.tolist(),

            z = w_x.Confirmed.values.tolist(),

            colorbar = {'title':"Count"}

            )

layout = dict(title = 'COVID-19 Positive Cases around the world',

              geo = dict(scope='world')

             )

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
df3 = pd.read_csv(path+'population_india_census2011.csv')

(df3.style

 .hide_index()

 .bar(color='#00FFFF', vmin=df3.Population.min(), subset=['Population'])

 .bar(color='#FF6F61', vmin=df3['Rural population'].min(), subset=['Rural population'])

 .bar(color='mediumspringgreen', vmin=df3['Urban population'].min(), subset=['Urban population'])

 .bar(color='orange', vmin=df3['Gender Ratio'].min(), subset=['Gender Ratio'])

 .set_caption(''))
df = pd.read_csv(path+'covid_19_india.csv')

df['country'] = 'India'

df.head()
data = dict(type='choropleth',

            locations = ['india'],

            locationmode = 'country names',

            colorscale = 'Reds',

            text = ['Total Cases'],

            z = [df.groupby('State/UnionTerritory')['Confirmed'].max().sum()],

#             colorbar = {'title':"Stores Count"}

            )

layout = dict(title = 'Total COVID-19 Positive Cases in India',

              geo = dict(scope='asia')

             )

choromap = go.Figure(data = [data],layout = layout)

iplot(choromap)
df.groupby('State/UnionTerritory')['Confirmed'].sum().iplot(kind = 'bar', xTitle = 'State/UnionTerritory', 

                                                              yTitle = 'Confirmed Cases', color ='tomato')
fig = px.bar(df, x="State/UnionTerritory", y="Confirmed", color='State/UnionTerritory', 

             color_continuous_scale=px.colors.sequential.Plasma, )

fig.show()
fig = px.line(df, x="Date", y="Confirmed", color='State/UnionTerritory')

fig.show()
fig = px.bar(df, x="Date", y="Confirmed", color='State/UnionTerritory')

fig.show()
fig = px.scatter(df, x="Date", y="Confirmed", color="State/UnionTerritory",

                 size='Confirmed', color_continuous_scale=px.colors.sequential.Plasma,)

fig.show()
fig = px.scatter(df, x='Date', y='Cured', color='State/UnionTerritory', title='Cases Cured').update_traces(mode='lines+markers')

fig.show()
df.groupby('State/UnionTerritory')['Cured'].sum().iplot(kind='bar', color='green', title='Cured Cases w.r.t Sate/UnionTerritory')
fig = px.scatter(df, x='Date', y='Deaths', color='State/UnionTerritory', title='Deaths occured').update_traces(mode='lines+markers')

fig.show()
df.groupby('State/UnionTerritory')['Deaths'].sum().iplot(kind='bar', color='red', title='Deaths Occured w.r.t Sate/UnionTerritory')
fig = px.scatter_matrix(df,

    dimensions=["Confirmed", "Cured"],

    color="State/UnionTerritory", title='Plot between Confirmed and Cured')

fig.show()

fig = px.scatter_matrix(df,

    dimensions=["Confirmed", "Deaths"],

    color="State/UnionTerritory", title='Plot between Confirmed and Deaths')

fig.show()
def div(x,y): 

    if y==0:

        return 0

    return x/y

df['mortality_rate'] = df.apply(lambda row: div(row['Deaths'], row['Confirmed']) , axis = 1)

fig = px.line(df, x="Date", y="mortality_rate", color='State/UnionTerritory')

fig.show()
df.groupby('State/UnionTerritory')['mortality_rate'].sum().iplot(kind='bar', color='darkblue')
corr = df[['Confirmed', 'Cured', 'Deaths']].corr()

ff.create_annotated_heatmap(

    z=corr.values,

    x=list(corr.columns),

    y=list(corr.index),

    annotation_text=corr.round(2).values,

    showscale=True, colorscale='emrld')
df2 = pd.read_csv(path+'IndividualDetails.csv')

df2.head()
x = df2.current_status.value_counts()

x = pd.DataFrame(data={'Current_satus': x.index.tolist(), 'Count': x.values.tolist()})

fig = px.pie(x, values='Count', names='Current_satus', title='Current Staus of COVID-19 Victims')

fig.show()
x = df2.notes.map(lambda x:str(x).title()).value_counts()[2:].head(20)

x = pd.DataFrame(data={'Current_satus': x.index.tolist(), 'Count': x.values.tolist()})

fig = px.pie(x, values='Count', names='Current_satus', title='Top 20 Reasons for COVID-19 Spread')

fig.show()
x = df2['gender'].value_counts()

x = pd.DataFrame(data={'Gender': x.index.tolist(), 'Count': x.values.tolist()})

fig = px.pie(x, values='Count', names='Gender', title='Who are affected by COVID-19(M/F)?')

fig.show()