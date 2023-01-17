# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

pokemon = pd.read_csv('../input/Pokemon.csv')

#heatmap
plt.subplots(figsize=(14, 10))
sns.heatmap(
    pokemon[pokemon['Type 2']!='None'].groupby(['Type 1', 'Type 2']).size().unstack(),
    linewidths=0,
    annot=True,
    cmap="rainbow"
)
plt.xticks(rotation=35)
plt.show()


type_to_int_dict = {
    'Grass': 0, 
    'Fire': 1, 
    'Water': 2, 
    'Bug': 3, 
    'Normal': 4, 
    'Poison': 5, 
    'Electric': 6, 
    'Ground': 7, 
    'Fairy': 8, 
    'Fighting': 9,
    'Psychic' : 10, 
    'Rock': 11, 
    'Ghost':12, 
    'Ice' : 13, 
    'Dragon': 14, 
    'Dark': 15, 
    'Steel': 16, 
    'Flying': 17
} 

#boxplot
pokemon['Type 1'] = pokemon['Type 1'].map(type_to_int_dict).astype(int)
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize=(12,6))
sns.boxplot(
    ax = ax, 
    x="Type 1", 
    y="Total", 
    data=pokemon, 
    palette="rainbow"
)
sns.despine(offset=5, trim=True)

pokemon['Atk - Def'] = pokemon['Attack'] - pokemon['Defense']
pokemon['Sp.Atk - Sp.Def'] = pokemon['Sp. Atk'] - pokemon['Sp. Def']
predictors = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed','Atk - Def','Sp.Atk - Sp.Def']
pk_mean = pokemon.groupby('Type 1').mean()[predictors]
data = pokemon[predictors]

#violinplot
f, ax = plt.subplots(figsize=(12, 6))
sns.violinplot(
    data=data, 
    palette="rainbow", 
    bw=.2, 
    cut=1, 
    linewidth=1
)


ax.set(ylim=(-120, 200))
ax.set_title("Important Features of Pokemon")
sns.despine(left=True, bottom=True)

