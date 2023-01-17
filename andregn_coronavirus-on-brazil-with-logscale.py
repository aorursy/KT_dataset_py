import numpy as np

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import plotly.express as px

import folium, json, math, requests, plotly

from branca.colormap import linear

from bs4 import BeautifulSoup
states = pd.read_csv('/kaggle/input/brazilianstates/states.csv')

states.head()
corona = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv')

corona = corona.merge(states, on='state')

corona['potentials'] = corona['population'] * 0.2

corona['potentials'] = corona['potentials'].astype(int)

corona['date'] = pd.to_datetime(corona['date'])

corona.head()
corona.dtypes
group_uf = corona.groupby('state')

uf = group_uf.tail(1).sort_values('cases', ascending=False).drop(columns=['date','hour']).set_index('state')

uf.style.background_gradient(cmap='Reds', subset=['suspects','refuses','cases','deaths'])
d = {'cases' : uf['cases'].sum(), 'deaths': uf['deaths'].sum()}

total = pd.DataFrame(d.items(), columns=['type', 'total_count']).set_index('type')

total
plt.figure(figsize=(10, 6))

bar = sns.barplot(x=total.index, y="total_count", data=total)

for p in bar.patches:

    bar.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),ha='center', va='bottom',color= 'black')

bar.set_title('Total count of cases on Brazil until {}'.format(corona.iloc[-1,0].strftime('%d/%m/%Y')), fontsize=16)

plt.xlabel('Cases types')

plt.ylabel('Cases counts')

plt.show()
cumulated = corona.groupby('date').sum()
ig, ax = plt.subplots(figsize=(16, 8))

sns.set(style="darkgrid")

sns.lineplot(x=cumulated.index, y='cases', data=cumulated, color='orange')

plt.yscale("log")

plt.xticks(rotation=90, ha='right')

plt.title('COVID-19 confirmed cases on Brazil until {}'.format(corona.iloc[-1,0].strftime('%d/%m/%Y')), fontsize=14)

plt.ylabel('Number of cases', fontsize=12)

plt.xlabel('Date', fontsize=12)

plt.axvline('2020-03-16', 0, 1200, c='dodgerblue', linestyle='--', linewidth=2, alpha=1, label='Companies start quarantine')

plt.axvline('2020-03-21', 0, 1200, c='k', linestyle='dotted', linewidth=2, alpha=1, label='SP government declares quarantine')

plt.legend(loc=2, fancybox=True, fontsize=10)

plt.show();
fig, ax = plt.subplots(figsize=(16, 8))

sns.set(style="darkgrid")



labels = []

for s in uf.index[:10]:

    l = sns.lineplot(x='date', y='cases', data=corona[(corona['state'] == s)])

    labels.append(s)

plt.ylabel('Casos Confirmados', fontsize=12)

plt.xlabel('Data', fontsize=12)

plt.xticks(rotation=90, ha='right')

plt.title('COVID-19 Casos Confirmados no Brasil Por Estados Até {} (em escala linear)'.format(corona.iloc[-1,0].strftime('%d/%m/%Y')), fontsize=14)



fig.legend(loc=6, bbox_to_anchor=(0.063, 0.4), labels=labels)

plt.legend(loc=2, fancybox=True, fontsize=10)

plt.xlim(xmin="2020-02-22")



plt.show();
fig, ax = plt.subplots(figsize=(16, 8))

sns.set(style="darkgrid")



labels = []

for s in uf.index[:10]:

    l = sns.lineplot(x='date', y='cases', data=corona[(corona['state'] == s)])

    labels.append(s)

plt.yscale("log")

plt.ylabel('Casos Confirmados', fontsize=12)

plt.xlabel('Data', fontsize=12)

plt.xticks(rotation=90, ha='right')

plt.title('COVID-19 Casos Confirmados no Brasil Por Estados Até {} (em escala logarítmica)'.format(corona.iloc[-1,0].strftime('%d/%m/%Y')), fontsize=14)



fig.legend(loc=6, bbox_to_anchor=(0.063, 0.4), labels=labels)

plt.legend(loc=2, fancybox=True, fontsize=10)

plt.xlim(xmin="2020-02-22")



plt.show();
for s in uf.index:

    ig, ax = plt.subplots(figsize=(20, 8))

    plt.xticks(rotation=90, ha='right')

    sns.lineplot(x='date', y='cases', data=corona[(corona['state'] == s)])

    sns.lineplot(x='date', y='suspects', data=corona[(corona['state'] == s)])

    plt.title('Number of cases on {}'.format(s), fontsize=14);

    plt.legend(labels=['confirmed','suspects'])

    plt.ylabel('Number of cases', fontsize=12)

    plt.xlabel('Date', fontsize=12)

    plt.show();
events = corona.groupby('date').sum()

events.tail()
events.sum()
ig, ax = plt.subplots(figsize=(12, 8))

bar = sns.barplot(x=uf['uf'], y=uf['cases'])

for p in bar.patches:

    bar.annotate('{:.0f}'.format(p.get_height()), (p.get_x()+0.4, p.get_height()),ha='center', va='bottom',color= 'black')

plt.ylabel('Total confirmed cases', fontsize=12)

plt.xlabel('States', fontsize=12)

plt.title('Number of confirmed cases by state until {}'.format(corona.iloc[-1,0].strftime('%d/%m/%Y')), fontsize=14)

plt.show();
url = '/kaggle/input/brazil-geojson/brazil_geo.json'

geo = json.load(open(url))
df = uf.reset_index().set_index('uf')

df.head()
colormap = linear.YlOrRd_09.scale(0,50)



map = folium.Map(

    width=800, height=600,

    location=[-15.77972, -47.92972], 

    zoom_start=4

)

folium.GeoJson(

    geo,

    name='cases',

    style_function=lambda feature: {

        'fillColor': colormap(df['cases'][feature['id']]),

        'color': 'black',

        'weight': 0.4,

    }

).add_to(map)

colormap.caption = 'Confirmed COVID-19 cases per state'

colormap.add_to(map)



map
weeks = corona.groupby('state').tail(7)

weeks['date'] = weeks['date'].dt.strftime('%d/%m/%Y')

weeks = weeks.pivot_table(index='uf', columns='date', values='suspects').fillna(0)

weeks.style.background_gradient(cmap='Reds')
ig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(weeks, linewidths=1, cmap="Blues")

plt.ylabel('States', fontsize=12)

plt.xlabel('Dates', fontsize=12)

plt.title('Suspect cases until {}'.format(corona.iloc[-1,0].strftime('%d/%m/%Y')), fontsize=16)

plt.show();
weeks = corona.groupby('state').tail(7)

weeks['date'] = weeks['date'].dt.strftime('%d/%m/%Y')

weeks = weeks.pivot_table(index='uf', columns='date', values='cases').fillna(0)

weeks.style.background_gradient(cmap='Reds')
ig, ax = plt.subplots(figsize=(10, 10))

sns.heatmap(weeks, linewidths=1, cmap="Blues")

plt.ylabel('States', fontsize=12)

plt.xlabel('Dates', fontsize=12)

plt.title('Confirmed cases until {}'.format(corona.iloc[-1,0].strftime('%d/%m/%Y')), fontsize=16)

plt.show();