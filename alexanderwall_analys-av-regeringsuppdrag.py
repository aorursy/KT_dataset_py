import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/cleaned_20180830.csv').drop('Unnamed: 0', axis=1) # importera skrapad data
df['date'] = pd.to_datetime(df['date']) # gör om datumsträng till Timestamp

df.head()
df.groupby(df['date'].map(lambda x: x.year))['title'].count().plot.bar(color='darkblue', title='Antal regeringsuppdrag per år')
df2 = df['tags'].str.split('|')
from collections import Counter

def flatten(l):
    res = []
    exclude = ['Regeringsuppdrag', 'Regeringen']
    for item in l:
        for subitem in item:
            if subitem not in exclude:
                res.append(subitem)
    return Counter(res).most_common()

tags_2015_2018 = flatten(df2[df['date'] >= '2015-01-01'].tolist())
df_2015_2018 = pd.DataFrame(tags_2015_2018, columns=['Tagg', 'Antal 2015-2018'])
df_2015_2018[:10]
tags_2011_2014 = flatten(df2[(df['date'] >= '2011-01-01') & (df['date'] <= '2014-12-31')].tolist())
df_2011_2014 = pd.DataFrame(tags_2011_2014, columns=['Tagg', 'Antal 2011-2014'])
df_2011_2014[:10]
df2 = df[df.date >= '2011-01-01'].groupby('target')['title'].count().sort_values(ascending=False).reset_index()[:10]
df2.columns = 'Mottagande myndighet','Antal uppdrag sedan 2011'
df2
df2 = df[(df['date'] >= '2011-01-01') & df['target'].isin(['Socialstyrelsen', 'Statskontoret', 'Statens skolverk', 'Kammarkollegiet', 'Boverket', 'Trafikverket'])].groupby(['target', df['date'].map(lambda x: x.year)])['title'].count().reset_index().sort_values(['date', 'title'], ascending=False)
df2.columns = ['Mottagande myndighet', 'År', 'Antal']
df2 = df2.set_index(['År', 'Mottagande myndighet']).unstack()
df2.columns = df2.columns.droplevel()
df2.plot()
df_dnr = df['dnr'].str.extract(r'(?P<ministry>[A-ZÅÄÖa-zåäö]{1,3})[\d]{4,5}\/[\d]{3,5}\/(?P<department>[A-ZÅÄÖa-zåäö]{1,5})')
df2 = pd.concat([df, df_dnr], axis=1)

df2[pd.notna(df2.ministry)].groupby(['ministry','department'])['title'].count().sort_values(ascending=False)[:10].plot.barh()
import warnings
warnings.filterwarnings('ignore')

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

df3 = df2[(df2['date'] >= '2017-01-01') & pd.notna(df2.ministry)]
df3.ministry = pd.Categorical(df3.ministry)
df3 = df3.reset_index(drop=True)

labels = {}

for index, row in df3.iterrows():
    labels[row['title']] = ''
    G.add_node(row['title'], group=row['ministry'])
    
for index, row in df3.iterrows():
    if row['department'] and row['ministry']:
        labels[row['department']] = row['department']
        G.add_node(row['department'], group=row['ministry'])
        
for index, row in df3.iterrows():
    if row['department'] and row['ministry']:
        G.add_edge(row['title'], row['department'])

plt.figure(figsize=(25,25))
options = {
    'edge_color': '#000000',
    'width': 1,
    'with_labels': True,
    'font_weight': 'regular',
    'font_size': 30,
    'font_color': 'darkred'
}

category_values = []
for node in G:
    category_values.append(G.node[node]['group'])

categories = list(set(category_values))
    
mapping = {categories[i]: i for i in range(0, len(categories))}
category_values_num = [mapping[x] for x in category_values]

nx.draw(G, labels=labels, node_color=category_values_num, cmap=plt.cm.Set1, **options)
