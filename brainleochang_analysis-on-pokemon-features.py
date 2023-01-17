import pandas as pd

import seaborn as sns

import matplotlib.pyplot as plt

import numpy as np



# read data

pokemon = pd.read_csv("../input/Pokemon.csv")

type1 = pokemon['Type 1'].unique()

len(type1)

pk_type1 = pokemon.groupby('Type 1').count()['#']
# take a look at the percentage of different types of pokemon

labels = 'Water', 'Normal', 'Grass', 'Bug', 'Psychic', 'Fire', 'Electric', 'Rock', 'Other'

sizes = [112, 98, 70, 69, 57, 52, 44, 44, 175]

colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral', 'yellow', 'lightgreen', 'silver', 'white', 'pink']

explode = (0, 0, 0, 0, 0, 0, 0, 0, 0.1)  # only "explode" the 2nd slice (i.e. 'Hogs')



plt.pie(sizes, explode=explode, labels=labels, colors=colors,

        autopct='%1.1f%%', shadow=True, startangle=90)

# Set aspect ratio to be equal so that pie is drawn as a circle.

plt.axis('equal')

plt.title("Percentage of Different Types of Pokemon")
# convert type1 to int

####### Thanks for Paul's comment, I changed this part to 'map' #######

type_to_int_dict = { 'Grass': 0, 'Fire': 1, 'Water': 2, 'Bug': 3, 'Normal': 4, 

                    'Poison': 5, 'Electric': 6, 'Ground': 7, 'Fairy': 8, 'Fighting': 9,

                    'Psychic' : 10, 'Rock': 11, 'Ghost':12, 'Ice' : 13, 'Dragon': 14, 

                    'Dark': 15, 'Steel': 16, 'Flying': 17} 

        

pokemon['Int_Type1'] = pokemon['Type 1'].map(type_to_int_dict).astype(int)



# Let's consider Total value of different Pokemon types

sns.set(style="ticks")

fig, ax = plt.subplots(figsize=(8,6))

sns.boxplot(ax = ax, x="Int_Type1", y="Total", data= pokemon, palette="PRGn")

sns.despine(offset=10, trim=True)

# seems difficult to determine type by total value
# Consider Pokemon features

pokemon['Atk - Def'] = pokemon['Attack'] - pokemon['Defense']

pokemon['Sp.Atk - Sp.Def'] = pokemon['Sp. Atk'] - pokemon['Sp. Def']

pk_mean = pokemon.groupby('Int_Type1').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed','Atk - Def','Sp.Atk - Sp.Def']]

predictors = ['HP','Speed','Atk - Def','Sp.Atk - Sp.Def']

data = pokemon[['HP','Speed','Atk - Def','Sp.Atk - Sp.Def']]

# distribution of these features for all pokemon

f, ax = plt.subplots(figsize=(8, 6))

sns.violinplot(data=data, palette="Set3", bw=.2, cut=1, linewidth=1)

ax.set(ylim=(-120, 200))

ax.set_title("Important Features of Pokemon")

sns.despine(left=True, bottom=True)
# distribution of HP among all types of pokemon

hp_data = pokemon[['Name','Type 1','HP']]

hp_data = hp_data.pivot_table(values = 'HP',index = ['Name'],  columns = ['Type 1'])

hp_data.head()

f, ax = plt.subplots(figsize=(18, 6))

sns.violinplot(data=hp_data, palette="Set3", bw=.2, cut=1, linewidth=1)

ax.set(ylim=(0, 200))

ax.set_title("HP of Different Types of Pokemon")

sns.despine(left=True, bottom=True)
# distributionof Speed among all types of pokemon

hp_data = pokemon[['Name','Type 1','Speed']]

hp_data = hp_data.pivot_table(values = 'Speed',index = ['Name'],  columns = ['Type 1'])

hp_data.head()

f, ax = plt.subplots(figsize=(18, 6))

sns.violinplot(data=hp_data, palette="Set3", bw=.2, cut=1, linewidth=1)

ax.set(ylim=(0, 200))

ax.set_title("Speed of Different Types of Pokemon")

sns.despine(left=True, bottom=True)
# distribution of Atk - Def among all types of pokemon

hp_data = pokemon[['Name','Type 1','Atk - Def']]

hp_data = hp_data.pivot_table(values = 'Atk - Def',index = ['Name'],  columns = ['Type 1'])

hp_data.head()

f, ax = plt.subplots(figsize=(18, 6))

sns.violinplot(data=hp_data, palette="Set3", bw=.2, cut=1, linewidth=1)

ax.set(ylim=(-150, 150))

ax.set_title("Atk - Def of Different Types of Pokemon")

sns.despine(left=True, bottom=True)
# distribution of Sp.Atk - Sp.Def among all types of pokemon

hp_data = pokemon[['Name','Type 1','Sp.Atk - Sp.Def']]

hp_data = hp_data.pivot_table(values = 'Sp.Atk - Sp.Def',index = ['Name'],  columns = ['Type 1'])

hp_data.head()

f, ax = plt.subplots(figsize=(18, 6))

sns.violinplot(data=hp_data, palette="Set3", bw=.2, cut=1, linewidth=1)

ax.set(ylim=(-150, 150))

ax.set_title("Sp.Atk - Sp.Def of Different Types of Pokemon")

sns.despine(left=True, bottom=True)
# Center the data to make it diverging

sns.set(style="white", context="talk")

f, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

data = pokemon.groupby("Type 1").mean()[['Speed', 'HP', 'Atk - Def', 'Sp.Atk - Sp.Def']]



sns.barplot(data.index, data['Atk - Def'], palette="RdBu_r", ax=ax1)

ax1.set_ylabel("Diverging in Atk - Def")



sns.barplot(data.index, data['Sp.Atk - Sp.Def'], palette="RdBu_r", ax=ax2)

ax2.set_ylabel("Diverging in Sp.Atk - Sp.Def")
# Cluster Analysis

# The cluster analysis code is learned from http://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html

import warnings

warnings.filterwarnings('ignore')

from sklearn.cluster import DBSCAN

from sklearn.preprocessing import StandardScaler

centers = [[1, 1], [-1, -1], [1, -1]]

data = pokemon[['Sp.Atk - Sp.Def', 'Atk - Def']]

data = StandardScaler().fit_transform(data)



# Compute DBSCAN

db = DBSCAN(eps=0.3, min_samples=10).fit(data)

core_samples_mask = np.zeros_like(db.labels_, dtype=bool)

core_samples_mask[db.core_sample_indices_] = True

labels = db.labels_



# Number of clusters in labels, ignoring noise if present.

n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)



# Plot result

import matplotlib.pyplot as plt



# Black removed and is used for noise instead.

unique_labels = set(labels)

colors = plt.cm.Spectral(np.linspace(0, 2, len(unique_labels)))

for k, col in zip(unique_labels, colors):

    if k == -1:

        # Black used for noise.

        col = 'k'



    class_member_mask = (labels == k)



    xy = data[class_member_mask & core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,

             markeredgecolor='k', markersize=14)



    xy = data[class_member_mask & ~core_samples_mask]

    plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,

             markeredgecolor='k', markersize=6)



plt.title('Estimated number of clusters: %d' % n_clusters_)

plt.show()

# the cluster result is not good, there are too many outliers and clusters are not obvious