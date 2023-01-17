%matplotlib nbagg



import matplotlib

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

import seaborn as sns
#pd.read_csv('../input/Pokemon.csv', index_col=0, encoding="ISO-8859-1")
data = pd.read_csv('../input/Pokemon.csv', index_col=0, encoding="ISO-8859-1")
# scatterplot

sns.lmplot(x='Attack', y='Defense', data=data)
# Scatterplot arguments

sns.lmplot(x='Attack', y='Defense', data=data,

           fit_reg=False, # No regression line

           hue='Stage')   # Color by evolution stage



# Tweak using Matplotlib

plt.ylim(0, None)

plt.xlim(0, None)
# Boxplot

sns.boxplot(data=data)
# using only combat stats

stats_data = data.drop(['Total', 'Stage', 'Legendary'], axis=1)



shw = sns.boxplot(data=stats_data)

shw = sns.swarmplot(data=stats_data, color=".25")
# violinplots are substitutes of box plot

# Set theme

sns.set_style('whitegrid')

 

# Violin plot

sns.violinplot(x='Type 1', y='Attack', data=data)
# color palette

pkmn_type_colors = ['#78C850',  # Grass

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



# Violin plot with Pokemon color palette

sns.violinplot(x='Type 1', y='Attack', data=data, 

               palette=pkmn_type_colors) # Set color palette

# Swarm plot with Pokemon color palette

sns.swarmplot(x='Type 1', y='Attack', data=data, 

              palette=pkmn_type_colors)

ax = sns.violinplot(x='Type 1', y='Attack', data=data, 

                    palette=pkmn_type_colors, inner=None)



ax = sns.swarmplot(x='Type 1', y='Attack', data=data,

                   color='k', alpha=0.7)

# Melt DataFrame

melted_data = pd.melt(stats_data, 

                    id_vars=["Name", "Type 1", "Type 2"], # Variables to keep

                    var_name="Stat") # Name of melted variable

print(melted_data.shape)
sns.swarmplot(x='Stat', y='value', data=melted_data, 

              hue='Type 1', palette=pkmn_type_colors,

              split=True)

plt.legend(bbox_to_anchor=(1, 1), loc=2)
# Calculate correlations

corr = stats_data.corr()

 

# Heatmap

sns.heatmap(corr)
# Distribution Plot (a.k.a. Histogram)

sns.distplot(data.Attack)
# Count Plot (a.k.a. Bar Plot)

sns.countplot(x='Type 1', data=data, palette=pkmn_type_colors)

 

# Rotate x-labels

plt.xticks(rotation=-45)
# Factor Plot (separate plots by categorical classes)

g = sns.factorplot(x='Type 1', 

                   y='Attack', 

                   data=data, 

                   hue='Stage',  # Color by stage

                   col='Stage',  # Separate by stage

                   kind='swarm') # Swarmplot

 

# Rotate x-axis labels

g.set_xticklabels(rotation=-45)

# Density Plot (distribution between two variables)

sns.kdeplot(data.Attack, data.Defense)