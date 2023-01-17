import numpy as np

import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import matplotlib.dates as mdates

import plotly.express as px

import folium, json, math, requests, plotly, warnings

from branca.colormap import linear

from bs4 import BeautifulSoup
print('Last update on', pd.to_datetime('now'))
warnings.filterwarnings('ignore')
states = pd.read_csv('../input/brazilianstates/states.csv')

states.columns = map(str.lower, states.columns)

states.head()
corona = pd.read_csv('/kaggle/input/corona-virus-brazil/brazil_covid19.csv').drop('region', axis=1)

corona = corona.merge(states, on='state', how='left')

corona['potentials'] = corona['population'] * 0.8

corona['potentials'] = corona['potentials'].astype(int)

corona['date'] = pd.to_datetime(corona['date'])

corona = corona[(corona['cases'] > 0)]

corona.head()
corona.dtypes
group_uf = corona.groupby('state')

uf = group_uf.tail(1).sort_values('cases', ascending=False).drop(columns=['date']).set_index('state')

uf.style.background_gradient(cmap='Reds', subset=['suspects','refuses','cases','deaths'])
d = {'cases' : uf['cases'].sum(), 'deaths': uf['deaths'].sum()}

total = pd.DataFrame(d.items(), columns=['type', 'total_count']).set_index('type')

total
fig, ax = plt.subplots(figsize=(12, 8))

plt.bar(x=total.index, height=total['total_count'], color=['limegreen','red'])



[ax.annotate('%s' % y, xy=(x-0.03,y+500), fontsize=14, fontweight='bold') for x,y in zip(range(0,2), total['total_count'])]

[ax.spines[side].set_visible(False) for side in ['left','right','top']]

plt.grid(which='major', axis='y')

plt.xlabel(None)

plt.ylabel('Cases count')

plt.xticks(fontsize=14)

plt.yticks(fontsize=12)

plt.title('COVID-19: number of cases in Brazil', fontsize=16, fontweight='bold', color='#333333')

plt.show();
cumulated = corona.groupby('date').sum().reset_index()

cumulated = cumulated[(cumulated['cases'] >= 100)]

cumulated['new_cases'] = cumulated['cases'].diff().fillna(0).astype(int)

cumulated['growth_cases'] = cumulated['cases'].diff().fillna(0).astype(int)/cumulated['cases']

cumulated['new_deaths'] = cumulated['deaths'].diff().fillna(0).astype(int)

cumulated['growth_deaths'] = cumulated['deaths'].diff().fillna(0).astype(int)/cumulated['deaths']

cumulated.head()
fig, ax = plt.subplots(figsize=(14, 10))

plt.plot(cumulated['date'], cumulated['cases'], color='limegreen', linewidth=8, alpha=0.5, marker='o')

plt.plot(cumulated['date'], cumulated['deaths'], color='red', linewidth=4, alpha=0.9, marker='o')

plt.bar(cumulated['date'], cumulated['new_cases'])

[ax.annotate('%s' % y, xy=(x,y+100), fontsize=10) for x,y in zip(cumulated['date'], cumulated['cases'])]



plt.xticks(rotation=90, ha='right')

plt.title('COVID-19: number of cases in Brazil', fontsize=18, fontweight='bold', color='#333333')



plt.ylabel('Number of cases', fontsize=12)

plt.xlabel(None)



plt.axvline('2020-03-16', 0, 1200, c='#CCCCCC', linestyle='--', linewidth=2, alpha=1)

ax.annotate('Companies start home-office', xy=('2020-03-16',19000), fontsize=12, rotation=90)

plt.axvline('2020-03-21', 0, 1200, c='#CCCCCC', linestyle='dotted', linewidth=2, alpha=1)

ax.annotate('SP government declares quarantine', xy=('2020-03-21',19000), fontsize=12, rotation=90)



plt.legend(loc=2, labels=['cases','deaths'], fontsize=14)



plt.grid(which='major', axis='y')

[ax.spines[side].set_visible(False) for side in ['left','right','top']]

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

plt.show();
fig, ax = plt.subplots(figsize=(14, 10))



threshold = 1



labels = []

for s in uf.index[:7]:

    plt.plot(corona['date'][(corona['state'] == s)], corona['cases'][(corona['state'] == s)], linewidth=4, alpha=0.9)

    labels.append(s)

    

plt.xticks(rotation=90, ha='right')

plt.title('COVID-19: number of cases per state in Brazil', fontsize=18, fontweight='bold', color='#333333')



plt.ylabel('Number of cases', fontsize=12)

plt.xlabel(None)



plt.legend(loc=6, fontsize=14, labels=labels)



plt.grid(which='major', axis='y', color='#EEEEEE')

[ax.spines[side].set_visible(False) for side in ['left','right','top']]

ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

plt.show();
plt.subplots(figsize=(40, 60))

for s, i in zip(uf.index, range(1,len(uf))):

    ax = plt.subplot(9,3,i)

    plt.subplots_adjust(bottom=-0.1)

    plt.xticks(rotation=90, ha='right', fontsize=16)

    plt.yticks(fontsize=16)

    

    c = corona[(corona['state'] == s)]

    plt.plot(c['date'], c['cases'], linewidth=8, color='limegreen', alpha=0.5, marker='o')

    plt.plot(c['date'], c['deaths'], linewidth=8, color='red', alpha=0.7, marker='o')



    ax.text(0.05,0.9,s, transform=ax.transAxes, fontsize=24, fontweight='bold')

    plt.ylabel(None)

    plt.xlabel(None)

    plt.legend(labels=['cases','deaths'], loc='center left', fontsize=16)

    plt.grid(which='major', axis='y')

    [ax.spines[side].set_visible(False) for side in ['left','right','top']]

    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m'))

    

plt.show();
fig, ax = plt.subplots(figsize=(12, 8))

plt.bar(uf['uf'], uf['cases'], color='limegreen')



[ax.annotate('%s' % y, xy=(x-0.5,y+100), fontsize=10, fontweight='bold') for x,y in zip(range(0,27), uf['cases'])]

[ax.spines[side].set_visible(False) for side in ['left','right','top']]

plt.grid(which='major', axis='y')

plt.ylim(0,6000)

plt.xlabel(None)

plt.ylabel('Cases count')

plt.xticks(fontsize=14)

plt.yticks(fontsize=12)

plt.title('COVID-19: number of cases per state in Brazil', fontsize=16, fontweight='bold', color='#333333')

plt.show();
url = '/kaggle/input/brazil-geojson/brazil_geo.json'

geo = json.load(open(url))
df = uf.reset_index().set_index('uf')

df.head()
fig, ax = plt.subplots(figsize=(12,10))

s = sns.scatterplot(data=df, x='deaths', y='cases', hue='region', s=300, alpha=0.5)

plt.legend(loc=5,markerscale=1.5, frameon=False, fontsize=12)

[ax.spines[side].set_visible(False) for side in ['left','right','top']]

plt.grid(which='major', axis='both', color='#EEEEEE')

plt.ylim(0,7000)

plt.xlabel('Deaths', fontsize=12)

plt.ylabel('Cases', fontsize=12)

plt.xticks(fontsize=12)

plt.yticks(fontsize=12)



d = df.reset_index().loc[:4,:]

for i in range(0,len(d)):

    plt.annotate(d.iloc[i]['uf'], xy=(d.iloc[i]['deaths']-3.5, d.iloc[i]['cases']+125))



plt.title('COVID-19: number of cases and deaths per state in Brazil', fontsize=16, fontweight='bold', color='#333333')

    

plt.show();
colormap = linear.YlOrRd_09.scale(0,5000)



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