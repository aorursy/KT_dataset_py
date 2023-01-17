# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.cluster import KMeans
import seaborn as sns; sns.set(style="ticks", color_codes=True, font_scale=2, rc={'figure.figsize':(15,9)})
import networkx as nx

from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
heroes_info = pd.read_csv('../input/heroes_information.csv')
heroes_info = heroes_info.drop(columns=heroes_info.columns[0])
powers = pd.read_csv('../input/super_hero_powers.csv')
heroes_info.replace('-', np.nan).info()
powers.replace('-', np.nan).info()
heroes_info = heroes_info.dropna()
heroes_info['Hair color'] = heroes_info['Hair color'].str.title()
heroes_info['Skin color'] = heroes_info['Skin color'].str.title()
heroes_info['Eye color'] = heroes_info['Eye color'].str.title()
heroes_info['Race'] = heroes_info['Race'].str.title()
heroes_info['Alignment'] = heroes_info['Alignment'].str.title()
heroes_info['Race'].sort_values().unique()
table = heroes_info['Race'].str.extract('(?P<Race>\w+)(.{1}[\-\/].{1}(?P<Race2>\w+))?', expand=True)[['Race', 'Race2']]
table['Race'] = table['Race'].fillna('no_race')
table['Race2'] = table['Race2'].fillna('no_race')
race_df = pd.concat([pd.get_dummies(table['Race']), pd.get_dummies(table['Race2']) ], axis=1).drop(columns='no_race')
heroes_info = heroes_info.drop(columns='Race')
heroes_info = pd.concat([heroes_info, race_df], axis=1)
heroes_info
heroes_info.loc[heroes_info['Gender'] != '-']['Gender'].value_counts().plot(kind='bar')
plt.title('Super heroes gender distribution')
plt.xticks(rotation=45)
heroes_info.loc[:, 'Alien':'Radiation'].sum().sort_values(ascending=False).head(20)
heroes_info.loc[:, 'Alien':'Radiation'].sum().sort_values(ascending=False).head(10).plot(kind='bar')
plt.title('Super heroes race distribution')
plt.xticks(rotation=45, ha='right')
sns.jointplot(x='Weight',y='Height',data=heroes_info,kind='kde',size=12)
# plt.title('Super heroes Height x Weight distribution')
heroes_info.loc[heroes_info['Eye color'] != '-']['Eye color'].value_counts().head(10).plot(kind='bar')
plt.title('Super heroes Eye Color distribution')
plt.xticks(rotation=45, ha='right')
heroes_info['Publisher'].value_counts().head(10).plot(kind='bar')
plt.title('Super heroes Publishers distribution')
plt.xticks(rotation=45, ha='right')
data = heroes_info.loc[heroes_info['Alignment'] != '-']
data['Alignment'].value_counts().plot(kind='bar')
plt.title('Super heroes Alignment distribution')
plt.xticks(rotation=45)
temp_series = heroes_info['Publisher'].value_counts().head(10)
labels = (np.array(temp_series.index))
sizes = (np.array((temp_series / temp_series.sum())*100))

temp_series.plot(kind='pie')
plt.title('Top 10 publishers with most heroes')
data = heroes_info.loc[heroes_info['Gender'] != '-']
data.groupby(['Publisher', 'Gender'])['Gender'].count().unstack('Gender').sort_values(by=['Male', 'Female'], ascending=False).head(10).plot(kind='bar', stacked=True)
plt.title('Super heroes Gender by Publisher')
plt.xticks(rotation=45, ha='right')
data = heroes_info.loc[heroes_info['Gender'] != '-']
percentual_data = data.groupby(['Publisher', 'Gender'])['Gender'].count() / data.groupby(['Publisher'])['Gender'].count()
percentual_data.unstack('Gender').sort_values(by=['Female', 'Male'], ascending=False).head(13).plot(kind='bar', stacked=True)
plt.title('Super heroes Gender percentual by Publisher')
plt.xticks(rotation=45, ha='right')
data = heroes_info.loc[(heroes_info['Gender'] != '-') & (heroes_info['Alignment'] != '-')]
percentual_data = data.groupby(['Alignment', 'Gender'])['Alignment'].count() / data.groupby(['Gender'])['Alignment'].count()
percentual_data.unstack('Alignment').plot(kind='bar', stacked=True)
plt.title('Super heroes Gender percentual by Alignment')
plt.xticks(rotation=45, ha='right')
data = heroes_info.loc[(heroes_info['Hair color'] != '-') & (heroes_info['Alignment'] != '-')]
data = data.groupby(['Alignment', 'Hair color'])['Alignment'].count()
data.unstack('Alignment').sort_values(by=['Good', 'Bad', 'Neutral'], ascending=False).head(12).plot(kind='bar')
plt.title('Super heroes Hair Color by Alignment')
plt.xticks(rotation=45, ha='right')
def function(group):
    return group.loc[:, 'Alien':'Radiation'].sum()
    
data = heroes_info.loc[heroes_info['Alignment'] != '-']
data = data.groupby('Alignment').apply(lambda group: function(group))
data.columns.name = 'Race'
#data_sort = data["Bad"] / data.sum(axis=0)
data_sort = data.loc["Bad"].divide(data.loc["Good"] + data.loc["Neutral"], fill_value=0).fillna(0).replace(np.inf, 0)
data = data.T
data['sort'] = data_sort
data.sort_values(by=['sort'], ascending=False).drop(columns='sort').head(15).plot(kind='bar')
plt.title('Super heroes Race by Alignment - sorted by higher Bad')
plt.xticks(rotation=45, ha='right')
powers
df_total_power = pd.concat([powers['hero_names'], pd.Series(powers.sum(axis=1), name='power')], axis=1)
df_total_power = df_total_power.sort_values('power', ascending=False)
df_total_power = df_total_power.merge(heroes_info, left_on='hero_names', right_on='name', how='inner').drop(columns='hero_names').set_index(['name', 'Publisher'])
df_total_power
df_total_power['power'].head(20).plot(kind='bar', figsize=(22,9))
plt.title('Most powerfull Superheroes')
plt.xticks(rotation=45, ha='right')
df_total_power.loc[df_total_power['Gender'] == 'Female']['power'].head(20).plot(kind='bar', figsize=(22,9))
plt.title('Most powerfull Female Superheroes')
plt.xticks(rotation=45, ha='right')
df_total_power.loc[df_total_power['Gender'] == 'Male']['power'].head(20).plot(kind='bar', figsize=(22,9))
plt.title('Most powerfull Male Superheroes')
plt.xticks(rotation=45, ha='right')
df_total_power.groupby('Publisher')['power'].sum().sort_values(ascending=False).head(8).plot(kind='bar')
plt.title('Combined Superheroes powers for each Publisher')
plt.xticks(rotation=45, ha='right')
data = df_total_power.loc[df_total_power['Gender'] != '-']

fig, ax1 = plt.subplots()
total = data.groupby('Gender')['power'].sum()
total.plot(kind='bar', ax=ax1)
ax1.set_ylabel('Total power')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
percentual = data.groupby('Gender')['power'].mean()
percentual.plot(kind='line', color='black', ax=ax2)
ax2.grid(False)
ax2.set_ylabel('Mean power')
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.title('Combined and mean power for each Gender Superheroes')
plt.xticks(rotation=45, ha='right')
heroes_info
s_info = heroes_info[['Gender', 'Eye color', 'Hair color', 'Publisher', 'Alignment']]
s_array = s_info.values.tolist()
s_array
oht = TransactionEncoder()
oht_ary = oht.fit(s_array).transform(s_array)
df = pd.DataFrame(oht_ary, columns=oht.columns_, index=s_info.index)
df = pd.concat([df, race_df == 1], axis=1)
df
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)
frequent_itemsets.sort_values(by=['support'], ascending=False)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
#association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)
rules = rules.loc[~rules['antecedents'].astype(str).str.contains('-')]
rules = rules.loc[~rules['consequents'].astype(str).str.contains('-')]
rules = rules.reset_index(drop=True)
rules
G=nx.from_pandas_edgelist(rules, 'antecedents', 'consequents', edge_attr=['antecedents', 'consequents'] )
plt.figure(1,figsize=(20,15))
plt.margins(0.2)
nx.draw(G, with_labels=True,node_size=1500, node_color="skyblue", node_shape="o", alpha=0.8, linewidths=6, font_size=23, font_color="black", font_weight="bold", width=1, edge_color="grey")
hero_power = powers.merge(heroes_info[['name', 'Alignment']], left_on='hero_names', right_on='name', how='inner').drop(columns=['hero_names'])
hero_power
hero_power = hero_power.loc[hero_power['Alignment'] != '-']
hero_names = hero_power['name']
hero_power = hero_power.drop(columns='name')
hero_power = pd.get_dummies(hero_power, columns=['Alignment'], prefix='', prefix_sep='')
hero_power = hero_power == 1
hero_power
frequent_itemsets = apriori(hero_power, min_support=0.25, use_colnames=True)
frequent_itemsets.sort_values(by=['support'], ascending=False)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
#rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.4)
rules = rules.loc[~rules['antecedents'].astype(str).str.contains('-')]
rules = rules.loc[~rules['consequents'].astype(str).str.contains('-')]
rules = rules.reset_index(drop=True)
rules
G=nx.from_pandas_edgelist(rules, 'antecedents', 'consequents', edge_attr=['antecedents', 'consequents'] )
plt.figure(1,figsize=(40,20))
plt.margins(0.4)
nx.draw(G, with_labels=True,node_size=1500, node_color="skyblue", node_shape="o", alpha=0.8, linewidths=6, font_size=23, font_color="black", font_weight="bold", width=1, edge_color="grey")
number_clusters = range(1, 10)
sum_of_squared_distances = []
df = hero_power
for k in number_clusters:
    kmeans = KMeans(n_clusters=k).fit(df)
    sum_of_squared_distances.append(kmeans.inertia_)
sns.set(font_scale=2)
plt.figure(1,figsize=(15,9))
plt.plot(number_clusters, sum_of_squared_distances, 'bx-')
plt.xlabel('number of clusters')
plt.ylabel('score')
plt.title('Elbow Method For Optimal k')
kmeans = KMeans(n_clusters=6).fit(df)
hero_power['cluster'] = kmeans.labels_
hero_power['name'] = hero_names
hero_clusters = hero_power[['name', 'cluster']].sort_values(by=['cluster'])
hero_clusters.to_csv('output.csv', index=False)
