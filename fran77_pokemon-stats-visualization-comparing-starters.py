# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns # graph

import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
data = pd.read_csv('../input/Pokemon.csv')
data.head()
len(data)
# How many Pokemon per generation ?

sns.set(rc={'figure.figsize':(9,7)})

sns.countplot(data.Generation)
data.groupby(['Generation'])['#'].count()
# Repartition of the Types

plt.figure(figsize=(10,8))

data.groupby(['Type 1'])['#'].count().plot.pie(autopct='%1.0f%%', pctdistance=0.9, labeldistance=1.1)
data.groupby(['Type 1'])['#'].count().sort_values(ascending=False)
# Water and Normal are the most recurrent types for the Pokemon
# Best Pokemon

sns.distplot(data.Total)

plt.axvline(data.Total.mean())
data.Total.sort_values(ascending=True)
# Who are the weakest Pokemon ?

data.iloc[data.Total.nsmallest(10).index.values][['Name', 'Total']]
# Who are the strongest ?

data.iloc[data.Total.nlargest(10).index.values][['Name', 'Total']]
# Which type is the better ?

g=sns.catplot(x='Type 1', y='Total', kind='bar', data=data)

g.set_xticklabels(rotation=90)
data.groupby(['Type 1'])['Total'].mean()
# Dragon is the best type
# Proportion of legendary pokemon

sns.countplot(data.Legendary)
sns.violinplot(x='Legendary', y='Total', data=data)
fig, ax = plt.subplots(figsize=(9, 7))

# Draw the two density plots

legendary = data[data.Legendary == True]

not_legendary = data[data.Legendary == False]



ax = sns.kdeplot(legendary.Attack, legendary.Defense,

                 cmap="Reds", shade=True, shade_lowest=False)

ax = sns.kdeplot(not_legendary.Attack, not_legendary.Defense,

                 cmap="Blues", shade=True, shade_lowest=False)



# Add labels to the plot

red = sns.color_palette("Reds")[-2]

blue = sns.color_palette("Blues")[-2]

ax.text(25, 0, "Non Legendary", size=16, color=blue)

ax.text(125, 150, "Legendary", size=16, color=red)
legend = pd.melt(data[['HP', 'Attack', 'Defense',

       'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']], "Legendary", var_name="attributes")

sns.swarmplot(x="attributes", y="value", hue="Legendary", data=legend)
# Difference between generations

sns.boxplot(x="Generation", y="Total", data=data)
# The 4th generation seems to be the best
generations = data.groupby('Generation')[['Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed', 'Legendary']].mean()
generations = generations.reset_index()
generations.index = generations.index + 1
generations
data.columns
sns.lineplot(data=generations[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']])
data[data.Name == "Pikachu"].index[0]
# Radar chart for Pokemon

%matplotlib inline



labels = np.array(['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed'])

pikachu = data.loc[data[data.Name == "Pikachu"].index[0],labels].values
pikachu
angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

pikachu=np.concatenate((pikachu,[pikachu[0]]))

angles=np.concatenate((angles,[angles[0]]))
fig = plt.figure(figsize=(20,10))

ax = fig.add_subplot(111, polar=True)

ax.plot(angles, pikachu, 'o-', linewidth=2)

ax.fill(angles, pikachu, alpha=0.25)

ax.set_thetagrids(angles * 180/np.pi, labels)

ax.set_title(data.loc[data[data.Name == "Pikachu"].index[0],"Name"])

ax.grid(True)
# Who is the best starter ? (1st Generation)

# Short term
data.iloc[:15]
bulbasaur = data.loc[data[data.Name == "Bulbasaur"].index[0],labels].values

charmander = data.loc[data[data.Name == "Charmander"].index[0],labels].values

squirtle = data.loc[data[data.Name == "Squirtle"].index[0],labels].values



angles=np.linspace(0, 2*np.pi, len(labels), endpoint=False)

# close the plot

bulbasaur = np.concatenate((bulbasaur,[bulbasaur[0]]))

charmander = np.concatenate((charmander,[charmander[0]]))

squirtle = np.concatenate((squirtle,[squirtle[0]]))



angles = np.concatenate((angles,[angles[0]]))
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(111, polar=True)

ax1.plot(angles, bulbasaur, 'o-', linewidth=2, label = 'Bulbasaur')

ax1.fill(angles, bulbasaur, alpha=0.25)

ax1.set_thetagrids(angles * 180/np.pi, labels)

ax1.grid(True)



ax2 = fig.add_subplot(111, polar=True)

ax2.plot(angles, charmander, 'o-', linewidth=2, label = 'Charmander')

ax2.fill(angles, charmander, alpha=0.25)

ax2.set_thetagrids(angles * 180/np.pi, labels)

ax2.grid(True)



ax3 = fig.add_subplot(111, polar=True)

ax3.plot(angles, squirtle, 'o-', linewidth=2, label = 'Squirtle')

ax3.fill(angles, squirtle, alpha=0.25)

ax3.set_thetagrids(angles * 180/np.pi, labels)

ax3.grid(True)



plt.legend(bbox_to_anchor=(0.9,1))
# In the long term : Final starters evolution
venusaur = data.loc[data[data.Name == "Venusaur"].index[0],labels].values

charizard = data.loc[data[data.Name == "Charizard"].index[0],labels].values

blastoise = data.loc[data[data.Name == "Blastoise"].index[0],labels].values



venusaur = np.concatenate((venusaur,[venusaur[0]]))

charizard = np.concatenate((charizard,[charizard[0]]))

blastoise = np.concatenate((blastoise,[blastoise[0]]))
fig = plt.figure(figsize=(20,10))

ax1 = fig.add_subplot(111, polar=True)

ax1.plot(angles, venusaur, 'o-', linewidth=2, label = 'Venusaur')

ax1.fill(angles, venusaur, alpha=0.25)

ax1.set_thetagrids(angles * 180/np.pi, labels)

ax1.grid(True)



ax2 = fig.add_subplot(111, polar=True)

ax2.plot(angles, charizard, 'o-', linewidth=2, label = 'Charizard')

ax2.fill(angles, charizard, alpha=0.25)

ax2.set_thetagrids(angles * 180/np.pi, labels)

ax2.grid(True)



ax3 = fig.add_subplot(111, polar=True)

ax3.plot(angles, blastoise, 'o-', linewidth=2, label = 'Blastoise')

ax3.fill(angles, blastoise, alpha=0.25)

ax3.set_thetagrids(angles * 180/np.pi, labels)

ax3.grid(True)



plt.legend(bbox_to_anchor=(1.1,1))
# Best type combination ?
data.isnull().sum()
combination = data.dropna()
combination = combination[['Type 1', 'Type 2', 'Total']]
types_total = combination.groupby(['Type 1', 'Type 2']).mean()
types_total = types_total.reset_index()

types_total.columns = ['Type 1', 'Type 2', 'Mean']
types_total.head()
sns.set(rc={'figure.figsize':(25,10)})

sns.barplot(x='Type 1', y='Mean', hue='Type 2', data=types_total)
types_total.sort_values(by='Mean', ascending=False).head()
data[(data['Type 1'] == "Ground") & (data['Type 2'] == "Fire")]['#'].count()
# Just one, we will change the selection
types_count = combination.groupby(['Type 1', 'Type 2']).count()

types_count = types_count.reset_index()
types_filter = pd.merge(types_total, types_count, on=['Type 1', 'Type 2'])
types_filter.Total.sum() / types_filter.groupby(['Type 1', 'Type 2']).count().sum()['Total']
types_filter = types_filter[types_filter.Total > 3]
sns.barplot(x='Type 1', y='Mean', hue='Type 2', data=types_filter)
types_filter.sort_values(by='Mean', ascending=False).head()