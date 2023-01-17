# importing libraries

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns
# reading the dataset and using the name column as the index

pokedata = pd.read_csv('../input/pokemon.csv', index_col = 'name')
# reading the first 5 data entries

print(pokedata.head())

print()

print(pokedata.shape)
# dataset contains the following columns

pokedata.columns
Abilities = pokedata.loc[:,'abilities':'against_water']

Abilities.head()
pokedata = pokedata.reset_index()

pokedata.head()

pokedata = pokedata[['name', 'classfication',"type1","type2",'hp',"attack","defense","sp_attack","sp_defense",

                     "speed","generation","is_legendary"]]

pokedata.columns = pokedata.columns.str.title()

pokedata.head()
# finding nan values, if any

pokedata.isna().sum()
# replacing the nan values in type 2 with None

pokedata.Type2.fillna('None', inplace = True)
# Capitalizing the values in type1 and type2 columns

pokedata.Type1 = pokedata.Type1.str.capitalize()

pokedata.Type2 = pokedata.Type2.str.capitalize()
print(pokedata.Generation.value_counts().sort_index())

pokedata.Generation.value_counts().sort_index().plot(kind = 'Bar').grid(True, axis= 'y')
pokedata.Type1.value_counts().plot(figsize = (10,10), kind = 'pie', autopct = '%1.1f%%', shadow = True)
pokedata.Type2.value_counts().plot(figsize = (10,10), kind = 'pie', autopct = '%1.1f%%', shadow = True)
sns.catplot(x = 'Type1', data = pokedata, kind = 'count', order = pokedata.Type1.value_counts().index,

            height = 5, aspect = 1.75)

plt.xticks(rotation = 30)

plt.grid(True, axis = 'y')

sns.catplot(x = 'Type2', data = pokedata, kind = 'count', order = pokedata.Type2.value_counts().index,

            height = 5, aspect = 1.75)

plt.xticks(rotation = 30)

plt.grid(True, axis = 'y')
print(pokedata.Is_Legendary.value_counts())

pokedata.Is_Legendary.value_counts().plot(kind = 'pie', autopct = '%1.1f%%', shadow = True, figsize = (7,10))
pd.crosstab(pokedata.Generation, pokedata.Is_Legendary).plot(kind= 'bar', figsize = (12,5)).grid(True, axis = 'y')
pokemon_stats_generation = pokedata.groupby("Generation").mean()

pokemon_stats_generation[['Hp', 'Attack', 'Defense', 'Sp_Attack', 'Sp_Defense', 'Speed']].plot(kind = 'line',

                                                                                               figsize = (12,5),

                                                                                              grid = True)
plt.figure(figsize = (15,7))

sns.heatmap(pokedata[pokedata['Type2']!='None'].groupby(['Type1', 'Type2']).size().unstack(),

            annot = True, cmap = 'Blues', linewidth = 2)

plt.xticks(rotation = 30)