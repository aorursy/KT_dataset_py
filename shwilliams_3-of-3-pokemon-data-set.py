'''
Sheryl Williams
3 of 3: Data Analysis of Pokemon data set
April 23, 2018
'''

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sn # data visualization
import matplotlib.pyplot as ml # data visualisation as well
%matplotlib inline
import warnings

sn.set(color_codes = True, style="white")
%matplotlib inline
warnings.filterwarnings("ignore")
pokemon = pd.read_csv("../input/Pokemon.csv", sep=",", header=0)
#type of symbol used to separate the value, header at 0 is the name of each column


pokemon_palette = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]

#dropping the '# Column' from the data set
pokemon = pokemon.drop(['#'],1)
#number of rows and columns in the data set
pokemon.shape
#To see the types of columns provided in the data set
pokemon.columns
#shows the first five values
pokemon.head()
#shows the last five values
pokemon.tail()
#Provides basic statistical analysis
pokemon.describe()
pokemon.corr()
ml.figure(figsize=(20,10)) 
sn.heatmap(pokemon.corr(),annot=True)
ml.figure(figsize = (15,5))
sn.countplot(x='Type 1', data=pokemon, palette=pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
sn.countplot(x='Type 2', data=pokemon)
sn.countplot(pokemon['Generation'])
pokemon['Generation'].value_counts()
avg = (166+165+160+121+106+82)/6
print(avg)
ml.figure(figsize = (15,5))
sn.boxplot(x = "Type 1", y = "Defense", data = pokemon, palette=pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize=(15,10))
sn.swarmplot(x='Type 1', y='Defense', data=pokemon, palette = pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
sn.boxplot(x = "Type 1", y = "Attack", data = pokemon, palette=pokemon_palette)
ml.figure(figsize=(15,10))
sn.swarmplot(x='Type 1', y='Attack', data=pokemon, palette = pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
sn.boxplot(x = "Type 1", y = "HP", data = pokemon, palette=pokemon_palette)
ml.figure(figsize=(15,10))
sn.swarmplot(x='Type 1', y='HP', data=pokemon, palette = pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
sn.boxplot(x = "Type 1", y = "Sp. Atk", data = pokemon, palette=pokemon_palette)
ml.figure(figsize=(15,10))
sn.swarmplot(x='Type 1', y='Sp. Atk', data=pokemon, palette = pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
sn.boxplot(x = "Type 1", y = "Sp. Def", data = pokemon, palette=pokemon_palette)
ml.figure(figsize=(15,10))
sn.swarmplot(x='Type 1', y='Sp. Def', data=pokemon, palette = pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
sn.boxplot(x = "Type 1", y = "Speed", data = pokemon, palette=pokemon_palette)
ml.figure(figsize=(15,10))
sn.swarmplot(x='Type 1', y='Speed', data=pokemon, palette = pokemon_palette)
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
sn.countplot(pokemon[pokemon['Legendary']==True]['Type 1'])
ml.xticks(rotation=-45)
pokemon[pokemon['Legendary']==True]['Type 1'].value_counts()
ml.figure(figsize = (20,15))
sn.barplot(x="Type 1", y="Attack", hue="Legendary", data=pokemon);
ml.xticks(rotation=-45)
ml.figure(figsize = (20,15))
sn.barplot(x="Type 1", y="Sp. Atk", hue="Legendary", data=pokemon);
ml.xticks(rotation=-45)
ml.figure(figsize = (15,5))
#ml.title('add title here')
sn.countplot(pokemon[pokemon['Legendary']==False]['Type 1'], palette=pokemon_palette)
pokemon[pokemon['Legendary']==False]['Type 1'].value_counts()
pokemon=pokemon.groupby(['Generation','Type 1']).count().reset_index()
pokemon=pokemon[['Generation','Type 1','Total']]
pokemon=pokemon.pivot('Generation','Type 1','Total')
pokemon[['Grass', 'Fire', 'Water', 'Bug', 'Normal', 'Poison', 'Electric', 'Ground', 'Fairy', 'Fighting', 'Psychic', 'Rock', 'Ghost', 'Ice', 'Dragon']].plot(color=pokemon_palette,marker='o')
ml.legend(bbox_to_anchor=(1, 1), loc=2)
fig=ml.gcf()
fig.set_size_inches(15,10)
ml.show()