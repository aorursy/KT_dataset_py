# Dependencies

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

from subprocess import check_output

import matplotlib.spyplot as plt

import networkx as nx
data = pd.read_csv('../input/Pokemon.csv')
data.head()
data.tail()
print(len(data.isnull().any()))

data.isnull().any()
data['Type 2'].replace(np.nan, '0', inplace=True)
data['Type 2'].head(10)
print("Number of pokemon are: " + str(data['Name'].nunique()))
pd.DataFrame(data['Name'].unique().tolist(), columns=['Pokemon'])
npoke_total = data.copy()
npoke_total.columns
npoke_total = pd.concat([npoke_total['Name'], data['Total']], axis=1)
sns.set()

plt.figure(figsize=(8,20))

ax = sns.barplot(x='Total',y='Name',data=npoke_total.sort_values(by='Total', ascending=False).head(25))

ax.set(xlabel='Overall', ylabel='Pokemon')

plt.show()
g = nx.Graph()
g = nx.from_pandas_dataframe(data,source='Name',target='Type 1')

print(nx.info(g))
plt.figure(figsize=(20, 20))

pos=nx.spring_layout(g, k=0.15)

nx.draw_networkx(g,pos,node_size=25, node_color='blue')

plt.show()
gen1 = data[data.Generation == 1]
types = gen1['Type 1']

explode = np.arange(len(types.unique())) * 0.01



colors = [

    'red',

    'blue',

    'yellow',

    'green'

]

types.value_counts().plot.pie(

    explode=explode,

    colors=colors,

    title="Percentage of Different Types of Pokemon",

    autopct='%1.1f%%',

    shadow=True,

    startangle=90,

    figsize=(9,9)

)

plt.tight_layout()

plt.show()
legendary_gen1 = gen1.groupby('Legendary').size()

print(legendary_gen1)
types = gen1['Legendary']

explode = np.arange(len(types.unique())) * 0.01



colors = [

    'yellow',

    'green'

]

types.value_counts().plot.pie(

    explode=explode,

    colors=colors,

    title="Percentage of Legendary to NoN-Legendary",

    autopct='%1.1f%%',

    shadow=True,

    startangle=90,

    figsize=(6,6)

)

plt.tight_layout()

plt.show()
types = data['Legendary']

explode = np.arange(len(types.unique())) * 0.01



colors = [

    'pink',

    'yellow'

]

types.value_counts().plot.pie(

    explode=explode,

    colors=colors,

    title="Percentage of Legendary to NoN-Legendary (entire set)",

    autopct='%1.1f%%',

    shadow=True,

    startangle=90,

    figsize=(7,7)

)

plt.tight_layout()

plt.show()
g = nx.from_pandas_dataframe(gen1,source='Name',target='Type 1')

print(nx.info(g))
nx.Graph()
plt.figure(figsize=(20, 20))

pos=nx.spring_layout(g, k=0.0319)

nx.draw_networkx(g,pos,node_size=805, node_color='pink', font_size=15)

plt.show()
sns.set()

plt.figure(figsize=(22,14))

ax = sns.boxplot(x='Total',y='Type 1',data=gen1)

ax.set(ylabel='Pokemon type', xlabel='Overall')

plt.show()

lx = sns.boxplot(x='Attack', y='Type 1', data=gen1)

lx.set(xlabel='Type of pokemon', ylabel='Attack strength')

plt.show()
# Checking out overall speed

sns.distplot(gen1['Speed'])

plt.show();
# i love char

from IPython.display import Image

Image(url='http://img08.deviantart.net/fb0c/i/2013/082/7/5/004_charmander_by_pklucario-d5z1g9v.png')