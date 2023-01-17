%matplotlib inline

from matplotlib import pyplot as plt

import pandas as pd

import numpy as np
# Seaborn for plotting and styling

import seaborn as sns
# Read dataset

df = pd.read_csv("../input/Pokemon.csv",index_col=0,encoding="latin-1")
#Show top 5 rows

df.head()
# Recommended way

sns.lmplot(x='Attack', y='Defense', data=df) 

sns.lmplot(x='Speed', y='Attack', data=df) 

sns.lmplot(x='Speed', y='Defense', data=df)

# Alternative way

#sns.lmplot(x=df.Attack, y=df.Defense)
# Scatterplot arguments

sns.lmplot(x='Attack', y='Defense', data=df,

           fit_reg=False, # No regression line

           hue='Stage')   # Color by evolution stage

sns.lmplot(x='Attack', y='Defense', data=df,

           fit_reg=False, # No regression line

           hue='Legendary')   # Color by Legendary

sns.lmplot(x='Attack', y='Defense', data=df,

           fit_reg=False, # No regression line

           hue='Type 1')
sns.lmplot(x='Attack', y='Defense', data=df,fit_reg=False,hue='Stage')

# Tweak using Matplotlib

plt.ylim(0, None)

plt.xlim(0, None)
# Boxplot

sns.boxplot(data=df)
# Pre-format DataFrame

stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)

stats_df.head()

 

# New boxplot using stats_df

sns.boxplot(data=stats_df)
#color customize

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

sns.violinplot(x='Type 1', y='Attack', data=df, 

               palette=pkmn_type_colors) # Set color palette
# Set theme

sns.set_style('whitegrid')

 

# Violin plot

sns.violinplot(x='Type 1', y='Attack', data=df)
sns.violinplot(x='Type 1', y='Defense', data=df)
sns.violinplot(x='Type 2', y='Attack', data=df)
sns.violinplot(x='Type 2', y='Defense', data=df)
# Swarm plot with Pokemon color palette

sns.swarmplot(x='Type 1', y='Attack', data=df, 

              palette=pkmn_type_colors)
# Set figure size with matplotlib

plt.figure(figsize=(10,6))

 

# Create plot

sns.violinplot(x='Type 1',

               y='Attack', 

               data=df, 

               inner=None, # Remove the bars inside the violins

               palette=pkmn_type_colors)

 

sns.swarmplot(x='Type 1', 

              y='Attack', 

              data=df, 

              color='k', # Make points black

              alpha=0.7) # and slightly transparent

 

# Set title with matplotlib

plt.title('Attack by Type')
stats_df.head()
# Melt DataFrame

melted_df = pd.melt(stats_df, 

                    id_vars=["Name", "Type 1", "Type 2"], # Variables to keep

                    var_name="Stat") # Name of melted variable

melted_df.head()
print( stats_df.shape )

print( melted_df.shape )
# Swarmplot with melted_df

sns.swarmplot(x='Stat', y='value', data=melted_df, hue='Type 1')
# 1. Enlarge the plot

plt.figure(figsize=(10,6))

 

sns.swarmplot(x='Stat', 

              y='value', 

              data=melted_df, 

              hue='Type 1', 

              split=True, # 2. Separate points by hue

              palette=pkmn_type_colors) # 3. Use Pokemon palette

 

# 4. Adjust the y-axis

plt.ylim(0, 260)

 

# 5. Place legend to the right

plt.legend(bbox_to_anchor=(1, 1), loc=2)
# Calculate correlations

corr = stats_df.corr()

 

# Heatmap

sns.heatmap(corr)
# Distribution Plot (a.k.a. Histogram)

sns.distplot(df.Attack)
#Bar PlotPython

# Count Plot (a.k.a. Bar Plot)

sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)

 

# Rotate x-labels

plt.xticks(rotation=-45)
# Factor Plot

g = sns.factorplot(x='Type 1', 

                   y='Attack', 

                   data=df, 

                   hue='Stage',  # Color by stage

                   col='Stage',  # Separate by stage

                   kind='swarm') # Swarmplot

 

# Rotate x-axis labels

g.set_xticklabels(rotation=-45)

 

# Doesn't work because only rotates last plot

# plt.xticks(rotation=-45)
sns.barplot(x="Type 1", y="Speed", hue="Stage", data=df)

#sns.barplot(x="Type 1", y="Attack", hue="Stage", data=df)

#sns.barplot(x="Type 1", y="Defense", hue="Stage", data=df)
# Density Plot

sns.kdeplot(df.Attack, df.Defense)
# Joint Distribution Plot

sns.jointplot(x='Attack', y='Defense', data=df)
sns.factorplot(x="Type 2", y="Attack", hue="Type 1",

               col="Legendary", data=df, kind="swarm");
sns.factorplot(x="Type 2", y="Defense", hue="Type 1",

               col="Legendary", data=df, kind="swarm");
sns.factorplot(x="Type 1", y="Attack", hue="Type 2",

               col="Legendary", data=df, kind="swarm");
sns.factorplot(x="Type 1", y="Defense", hue="Type 2",

               col="Legendary", data=df, kind="swarm");