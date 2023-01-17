import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns
data = pd.read_csv ('../input/pokemon/pokemon.csv')

data.head()
#create a dataset for the number of pokemon in each generation

generations = pd.DataFrame ({'count': data.generation.value_counts().sort_index()})

generations 
#plot a bar chart

plt.figure (figsize = (20, 8))

sns.barplot (x = generations.index, y = generations['count'])

plt.title ('Number of Pokemon in Each Generation', fontdict = {'fontsize': 30})

plt.xlabel ('Generation', fontdict = {'fontsize': 20})

plt.ylabel ('Number of pokemon', fontdict = {'fontsize': 20})

plt.show()
#create a dataset that contains a row for each type each pokemon has

data2 = data[data.type2.notna()].copy()

data2.type1 = data2.type2

type_data = pd.concat ([data, data2])



#get a count for each type

type_count = type_data.type1.value_counts()

type_count = pd.DataFrame ({'count': type_count})

type_count
#plot a bar chart

plt.figure (figsize = (20, 8))

sns.barplot (x = type_count.index, y = type_count['count'])

plt.title ('Occurrences of pokemon types', fontdict = {'fontsize': 30})

plt.xlabel ('Type', fontdict = {'fontsize': 20})

plt.ylabel ('Number of pokemon with type', fontdict = {'fontsize': 20})

plt.show()
#create a dataset that has totals for each pokemon type in each generation

generations_types = type_data.groupby (['generation']).type1.value_counts()

generations_types = pd.DataFrame ({'count': generations_types})

generations_types['generation'] = generations_types.index.get_level_values(0)

generations_types['type'] = generations_types.index.get_level_values(1)

generations_types.reset_index(drop = True, inplace = True)

generations_types
#plot a line graph

plt.figure (figsize = (20, 8))

sns.lineplot (x = generations_types['generation'], y = generations_types['count'], hue = generations_types['type'])

plt.title ('Number of Pokemon of Each Type in Each Generation', fontdict = {'fontsize': 30})

plt.xlabel ('Generation', fontdict = {'fontsize': 20})

plt.ylabel ('Total pokemon', fontdict = {'fontsize': 20})

plt.show()
#create a data frame that contains counts of how many times each type combination occurs

type_combos = data.groupby (['type1', 'type2']).size().unstack()

for col in type_combos.columns:

    type_combos[col] = type_combos[col].fillna(0)

    

type_combos
#plot a heatmap

plt.figure (figsize = (20, 8))

sns.heatmap (data = type_combos, annot = True, linewidths = 1, cmap = 'Purples')

plt.title ('Pokemon Type Combinations', fontdict = {'fontsize': 30})

plt.xlabel ('Type 1', fontdict = {'fontsize': 20})

plt.ylabel ('Type 2', fontdict = {'fontsize': 20})

plt.show()
#plot a scatterplot

plt.figure (figsize = (20, 10))

sns.regplot (x = data.attack, y = data.defense)

plt.title ('Relationship Between Attack and Defence', fontdict = {'fontsize': 30})

plt.xlabel ('Attack', fontdict = {'fontsize': 20})

plt.ylabel ('Defence', fontdict = {'fontsize': 20})

plt.show()
#plot a scatterplot that also shows generation

sns.lmplot (data = data, x = 'attack', y = 'defense', hue = 'generation')

plt.title ('Relationship Between Attack and Defence in Each Generation', fontdict = {'fontsize': 20})

plt.xlabel ('Attack', fontdict = {'fontsize': 20})

plt.ylabel ('Defence', fontdict = {'fontsize': 20})

plt.show()
#plot a histogram

plt.figure (figsize = (20, 10))

sns.distplot (a = data.weight_kg, kde = False)

plt.title ('Distribution of Weight', fontdict = {'fontsize': 30})

plt.xlabel ('Weight (kg)', fontdict = {'fontsize': 20})

plt.ylabel ('Number of pokemon', fontdict = {'fontsize': 20})

plt.show()

#help (plt.title)