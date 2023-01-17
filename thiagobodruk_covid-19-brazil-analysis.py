import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

import plotly.express as px

import plotly.graph_objects as go

from urllib.request import urlopen

from bs4 import BeautifulSoup

from sklearn.preprocessing import StandardScaler, MinMaxScaler

from sklearn.cluster import KMeans, AffinityPropagation

import json, requests
print('Last update on', pd.to_datetime('now'))
def load_data(url):

    return pd.read_csv(url)
df = load_data('../input/corona-virus-brazil/brazil_covid19.csv').drop(columns=['region']).rename(columns={'date':'Date','state':'State','cases':'Cases','deaths':'Deaths'})

brazil = load_data('../input/brazilianstates/states.csv').rename({'Demographic Density': 'Density', 'Cities count': 'Cities'}, axis=1)

airports = load_data('../input/brazilianstates/airports.csv').rename({'Passengers rate': 'Airport'}, axis=1)

icu = load_data('../input/brazilianstates/icu-beds.csv').rename({'Public beds per citizen': 'Public BPC', 'Private beds per citizen': 'Private BPC'}, axis=1)

df = df.merge(brazil, how='left', on='State')

df = df.merge(airports, how='left', on='UF')

df = df.merge(icu, how='left', on='UF')

df.head()
def subnotif(row):

    d = row['Mortality']

    if d > 0.03: return ((0.97 * row['Deaths']) / 0.034) - row['Cases']

    else: return 0



    

def engineer_data(df):

    df['Date'] = pd.to_datetime(df['Date'])

    df['Mortality'] = df['Deaths']/df['Cases']

    df['Mortality'] = df['Mortality'].fillna(0)

    df['Subnotif'] = df.apply(subnotif, axis=1).astype(int)

    df['Cases per 1K'] = df['Cases']/(df['Population']/1000)

    df['Deaths per 1K'] = df['Deaths']/(df['Population']/1000)

    df = df.fillna(0)

    df = df[['Date', 'UF', 'State','Region', 'Capital', 'Cases', 'Deaths', 'Cases per 1K', 'Deaths per 1K', 'Mortality', 'Subnotif', 'Area','Cities',

             'Population', 'Density', 'Airport', 'GDP','GDP rate', 'Poverty','Latitude', 'Longitude','ICU beds',

             'Public beds','Private beds','Public BPC','Private BPC']]

    return df



df = engineer_data(df)

df.head()
pd.options.mode.chained_assignment = None

g = df.groupby('UF')

t = g.tail(1).sort_values('Cases', ascending=False).set_index('UF').drop(columns=['Date'])

t['Subnotif'][(t['Subnotif'] < 0)] = 0

t
c = df[['Date','Cases','Deaths']].groupby('Date').sum().reset_index()

c = c[(c['Cases'] >= 100)].melt(id_vars='Date', value_vars=['Cases', 'Deaths'])



fig = px.line(c, x='Date', y='value', color='variable')

fig.update_layout(title='COVID-19 in Brazil: total number of cases over time',

                  xaxis_title='Brazilian states', yaxis_title='Number of cases',legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.02,y=0.98))

fig.show()
s = df[(df['UF'].isin([i for i in t.index[:7]]))]

s = s[(s['Cases'] >= 100)]



fig = px.line(s, x='Date', y='Cases', color='UF')

fig.update_layout(title='COVID-19 in Brazil: total number of cases over time',

                  xaxis_title='Date', yaxis_title='Number of cases', legend_title='<b>Top 7 states</b>',

                  legend=dict(x=0.02,y=0.98))

fig.show()
fig = px.scatter(t, x="Population", y="Cases", title="COVID-19 in Brazil: population size vs number of cases by state",

                 size="Cases", color="Mortality",hover_name=t.index, log_x=True, log_y=True, size_max=60)

fig.show()
fig = px.scatter(t, x="Density", y="Cases", title="COVID-19 in Brazil: demographic density vs number of cases by state",

                 size="Cases", color="Mortality",hover_name=t.index, log_x=True, log_y=True, size_max=60)

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Cases', x=t.index, y=t['Cases']),

    go.Bar(name='Deaths', x=t.index, y=t['Deaths'])

])

fig.update_layout(barmode='stack', title="COVID-19 in Brazil: number of cases by state", 

                  xaxis_title="Brazilian states", yaxis_title="Number of cases", legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.84,y=0.5))

fig.show()
fig = px.scatter(t, x="Deaths", y="Cases", title="COVID-19 in Brazil: cases vs deaths by state",

                 size="Cases", color="Mortality",hover_name=t.index, log_y=True, log_x=True, size_max=60)

fig.show()
y = t.copy()

y.sort_values('Cases per 1K', ascending=False, inplace=True)

fig = go.Figure(data=[

    go.Bar(name='Cases per 1K citizens', x=y.index, y=y['Cases per 1K']),

    go.Bar(name='Deaths per 1K citizens', x=y.index, y=y['Deaths per 1K'])

])

fig.update_layout(barmode='stack', title="COVID-19 in Brazil: cases and deaths per 1K citizens by state", 

                  xaxis_title="Brazilian states", yaxis_title="Cases per 1K citizens", legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.78,y=0.5))

fig.show()
fig = go.Figure(data=[

    go.Bar(x=t.sort_values('Airport', ascending=False).index, y=t.sort_values('Airport', ascending=False)['Airport'],

    text=t.sort_values('Airport', ascending=False)['Airport'],

    marker_color='#EF553B', name='Airport traffic'),

    go.Scatter(x=t.sort_values('Airport', ascending=False).index,

           y=np.full((1,len(t.index)), t['Airport'].mean()).tolist()[0], marker_color='blue', name='Brazilian avg')

])



fig.update_layout(title='Airport traffic by state',

                  xaxis_title='Brazilian states', yaxis_title='Airport traffic', legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.75,y=0.95))

fig.show();
fig = px.scatter(t, x="Airport", y="Cases", title="COVID-19 in Brazil: airport traffic vs number of cases by state",

                 size="Cases", color="Mortality",hover_name=t.index, log_x=True, log_y=True, size_max=60)

fig.show()
fig = go.Figure(data=[

    go.Bar(x=t.sort_values('Mortality', ascending=False).index,

    y=t.sort_values('Mortality', ascending=False)['Mortality'],

    text=round(t.sort_values('Mortality', ascending=False)['Mortality'], 2),

    marker_color='#EF553B', name='Mortality'),

    go.Scatter(x=t.sort_values('Mortality', ascending=False).index,

           y=np.full((1,len(t.index)), 0.03).tolist()[0], marker_color='blue', name='World avg')

])



fig.update_layout(title='COVID-19 in Brazil: average mortality by state',

                  xaxis_title='Brazilian states', yaxis_title='Deaths per case', legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.75,y=0.5))

fig.show();
s = df[['Date','Cases','Subnotif']].groupby('Date').sum().reset_index()

s = s.rename({'Cases':'Confirmed cases','Subnotif':'Sub-notified cases'}, axis=1)

s = s[(s['Confirmed cases'] >= 100)].melt(id_vars='Date', value_vars=['Confirmed cases', 'Sub-notified cases'])



fig = px.area(s, x='Date', y='value', color='variable')

fig.update_layout(title='COVID-19 in Brazil: total number of cases vs sub-notifications',

                  xaxis_title='Brazilian states', yaxis_title='Number of cases',legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.02,y=0.98))

fig.show()
fig = go.Figure(data=[

    go.Bar(name='Cases', x=t.index, y=t['Cases']),

    go.Bar(name='Sub-notifications', x=t.index, y=t['Subnotif'])

])

fig.update_layout(barmode='stack', title="COVID-19 in Brazil: cases vs sub-notified cases by state", 

                  xaxis_title="Brazilian states", yaxis_title="Number of cases", legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.75,y=0.5))

fig.show()
fig = go.Figure(data=[

    go.Bar(x=t.sort_values('Public BPC', ascending=False).index, y=t.sort_values('Public BPC', ascending=False)['Public BPC'],

    text=t.sort_values('Public BPC', ascending=False)['Public BPC'],

    marker_color='#EF553B', name='ICU beds per citizen'),

    go.Scatter(x=t.sort_values('Public BPC', ascending=False).index,

           y=np.full((1,len(t.index)), t['Public BPC'].mean()).tolist()[0], marker_color='blue', name='Brazilian avg')

])



fig.update_layout(title='Number of ICU beds per 10K citizen by state',

                  xaxis_title='Brazilian states', yaxis_title='ICU beds per 10K citizen', legend_title='<b>COVID-19</b>',

                  legend=dict(x=0.75,y=0.95))

fig.show();
fig = px.scatter(t, x="Cases", y="Public BPC", title="Number of ICU beds per 10K citizens vs population size",

                 size="Cases", color="Mortality",hover_name=t.index, log_y=False, log_x=True, size_max=60)

fig.show()
h = df[(df['Cases'] >= 100)].sort_values(['Date','UF'])

fig = go.Figure(data=go.Heatmap(

        z=h['Cases'],

        x=h['Date'],

        y=h['UF'],

        colorscale='Viridis'))



fig.update_layout(

    title='COVID-19 in Brazil: number of cases over time', xaxis_nticks=45)



fig.show()
fig = go.Figure(data=go.Heatmap(

        z=h['Mortality'],

        x=h['Date'],

        y=h['UF'],

        colorscale='Viridis'))



fig.update_layout(

    title='COVID-19 in Brazil: mortality rate over time', xaxis_nticks=45)



fig.show()
geo = json.load(open('/kaggle/input/brazil-geojson/brazil_geo.json'))

fig = px.choropleth(t, geojson=geo, locations=t.index, color='Mortality',

            scope="south america",labels={'Cases':'Cases'})

fig.update_layout(title='COVID-19 in Brazil: cases by state', margin={"r":0,"t":0,"l":0,"b":0})

fig.show()
h = t.reset_index()

fig = px.treemap(h, path=['Region','UF'], values='Cases', color='Mortality', color_continuous_scale='thermal')

fig.update_layout(title='COVID-19 in Brazil: number of cases by region')

fig.show()
h = t.reset_index()

fig = px.treemap(h, path=['Region','UF'], values=t['Cases']/(t['Population']/1000), color='Mortality', color_continuous_scale='thermal')

fig.update_layout(title='COVID-19 in Brazil: cases per 1K citizens by region')

fig.show()
fig = px.scatter(t, x="Poverty", y="Mortality", title="COVID-19 in Brazil: poverty vs deaths by state",

                 size="Cases", color="Mortality",hover_name=t.index, log_x=False, log_y=False, size_max=60)

fig.show()
t.replace({'North':0, 'Northeast': 1, 'Center-west': 2, 'South': 3, 'Southeast': 4}, inplace=True)



fig, ax = plt.subplots(figsize=(12,10))

sns.heatmap(t.corr(), vmin=0, cmap='YlGnBu')

plt.show();
corr = t.corr().iloc[[0,1]].transpose()

corr = corr[(corr['Cases'] > 0.25)].sort_values('Cases', ascending=False)

features = corr.index.tolist()

features.append('Mortality')

features.remove('Private BPC')

print('Selected features:', features)
d = t[features].copy()

d.head()
def create_bins(df, columns, q=5):

    for column in columns:

        df[column] = pd.qcut(df[column], q, duplicates='drop').cat.codes



def normalize_data(df, columns):

    minMaxScaler = MinMaxScaler()

    df[columns] = minMaxScaler.fit_transform(d[columns])
create_bins(d, ['Airport', 'Subnotif', 'ICU beds', 'Private beds', 'Population', 'Public beds', 'Cities', 'GDP rate', 'GDP', 'Density'], q=7)

normalize_data(d, d.columns)

d.head()
kmeans = KMeans(n_clusters=5)

pred = kmeans.fit_predict(d[d.columns])

t['K-means'], d['K-means'] = [pred, pred]

d[d.columns].sort_values(['K-means','Cases','Deaths'], ascending=False).style.background_gradient(cmap='YlGnBu', low=0, high=0.2)
fig = px.treemap(t.reset_index(), path=['K-means','UF'], values='Cases')

fig.update_layout(title='K-means clusters')

fig.show()
c = t.sort_values(['K-means','Cases'], ascending=False)

data = [go.Bar(x=c[(c['K-means'] == i)].index, y=c[(c['K-means'] == i)]['Cases'],text=c[(c['K-means'] == i)]['Cases'],name=i) for i in range(0,5)]

fig = go.Figure(data=data)



fig.update_layout(title='K-means Clustering: number of cases by cluster',

                  xaxis_title='Brazilian states', yaxis_title='Deaths per case', legend_title='<b>Clusters</b>',

                  legend=dict(x=0.02,y=0.5))

fig.show();
c = t.sort_values(['K-means','Mortality'], ascending=False)

data = [go.Bar(x=c[(c['K-means'] == i)].index, y=c[(c['K-means'] == i)]['Mortality'],text=c[(c['K-means'] == i)]['Mortality'],name=i) for i in range(0,5)]

data.append(go.Scatter(x=t.sort_values('Mortality', ascending=False).index,

           y=np.full((1,len(t.index)), 0.03).tolist()[0], marker_color='black', name='World avg'))



fig = go.Figure(data=data)

fig.update_layout(title='K-means Clustering: mortality rate by cluster',

                  xaxis_title='Brazilian states', yaxis_title='Deaths per case', legend_title='<b>Clusters</b>',

                  legend=dict(x=0.82,y=0.5))

fig.show();