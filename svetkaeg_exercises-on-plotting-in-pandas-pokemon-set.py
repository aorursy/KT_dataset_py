import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/pokemon.csv")

pokemon.head()
pokemon.type1.value_counts().plot.bar()
pokemon.hp.value_counts().sort_index().plot.line()
pokemon.weight_kg.plot.hist()
pokemon.plot.scatter(x='defense', y='attack')
pokemon.plot.hexbin(x='defense', y='attack', gridsize=15)
pokemon_stats_legendary = pokemon.groupby(['is_legendary', 'generation']).mean()[['attack', 'defense']]
pokemon_stats_legendary.plot.bar(stacked=True)
pokemon_stats_by_generation = pokemon.groupby('generation').mean()[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]
pokemon_stats_by_generation.plot.line()
pokemon.plot.scatter(x='defense', 

                     y='attack', 

                     figsize=(12,6))
my_plot = pokemon.base_total.plot.hist(figsize=(12,6),

                             color='gray',

                             fontsize=16,

                             bins=50)



my_plot.set_title('Density by base total', fontsize=20)
my_plot = pokemon.type1.value_counts().plot.bar(figsize=(12,6),

                                                fontsize=14)



my_plot.set_title('Pokemon by primary type', fontsize=20)

sns.despine(bottom=True,

            left=True)
fig, axarr = plt.subplots(2, 1, figsize=(8,8))
fig, axarr = plt.subplots(2, 1, figsize=(8,8))

pokemon.defense.plot.hist(ax=axarr[0], bins=40)

axarr[0].set_title('Defense frequency', fontsize=14)

pokemon.attack.plot.hist(ax=axarr[1], bins=40)

axarr[1].set_title('Attack frequency', fontsize=14)
sns.countplot(pokemon.generation)
sns.distplot(pokemon.hp)
sns.jointplot(x=pokemon.attack, y=pokemon.defense)
sns.jointplot(x=pokemon.attack, y=pokemon.defense, kind='hex')
sns.kdeplot(pokemon.hp, pokemon.attack, shade=True)
sns.boxplot(x=pokemon.is_legendary, y=pokemon.attack)
sns.violinplot(x=pokemon.is_legendary, y=pokemon.attack)
grid = sns.FacetGrid(pokemon, row='is_legendary')

grid.map(sns.kdeplot, 'attack')
grid = sns.FacetGrid(pokemon, col='is_legendary', row='generation')

grid.map(sns.kdeplot, 'attack')
sns.pairplot(pokemon[['hp', 'attack', 'defense']])
sns.lmplot(x='attack', y='defense', hue='is_legendary', fit_reg=False, data=pokemon, markers = ['x', 'o'])
sns.boxplot(x='generation', y='base_total', hue='is_legendary', data=pokemon)
sns.heatmap(pokemon.loc[:,['hp', 'attack', 'sp_attack', 'defense', 'sp_defense', 'speed']].corr(), annot=True)
from pandas.plotting import parallel_coordinates

df = pokemon[pokemon.type1.isin(['fighting', 'psychic'])].loc[:, ['type1', 'attack', 'sp_attack', 'defense', 'sp_defense']]

parallel_coordinates(df, 'type1', colors=['violet', 'blue'])