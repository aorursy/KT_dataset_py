# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



# Pandas for managing datasets

import pandas as pd

# Matplotlib for additional customization

from matplotlib import pyplot as plt

# include the following command to get the output of the plots within the notebook, just below the plot code

%matplotlib inline

# Seaborn for plotting and styling

import seaborn as sns



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
# Read dataset

df = pd.read_csv('../input/Pokemon.csv', index_col=0)
# Display first 5 observations

df.head()
# Change the size of figures



sns.set(rc={'figure.figsize':(11.7,8.27)})



# alternatelively change them from matplotlib

from matplotlib import rcParams



# figure size in inches

rcParams['figure.figsize'] = 11.7,8.27
# Recommended way

sns.lmplot(x='Attack', y='Defense', data=df);

 

# Alternative way

# sns.lmplot(x=df.Attack, y=df.Defense)
sns.lmplot(x='Attack', y='Defense', data=df,

           fit_reg=False, # No regression line

           hue='Stage');   # Color by evolution stage
# Plot using Seaborn

sns.lmplot(x='Attack', y='Defense', data=df,

           fit_reg=False, 

           hue='Stage')

 

# Tweak using Matplotlib (limits has to be set with it)

plt.ylim(0, 200)

plt.xlim(0, 200);
# default Boxplot

sns.boxplot(data=df);
# Pre-format DataFrame

# We can remove the Total since we have individual stats.

# We can remove the Stage and Legendary columns because they aren't combat stats.

stats_df = df.drop(['Total', 'Stage', 'Legendary'], axis=1)

 

# New boxplot using stats_df

sns.boxplot(data=stats_df);
# Set theme

sns.set_style('whitegrid') # alternative here is 'darkgrid'

 

# Violin plot

sns.violinplot(x='Type 1', y='Attack', data=df);



#As you can see below, Dragon types tend to have higher Attack stats than Ghost types, but they also have greater variance.
# define a color palette with custom colors

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

# themes taken from http://bulbapedia.bulbagarden.net/wiki/Category:Type_color_templates



# Violin plot with Pokemon color palette

sns.violinplot(x='Type 1', y='Attack', data=df, 

               palette=pkmn_type_colors); # Set color palette
#Violin plots are great for visualizing distributions. However, since we only have 151 Pokémon in our dataset,

#we may want to simply display each point.

#That's where the swarm plot comes in. This visualization will show each point, while "stacking" those with similar values:



# Swarm plot with Pokemon color palette

sns.swarmplot(x='Type 1', y='Attack', data=df, 

              palette=pkmn_type_colors);
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

plt.title('Attack by Type');
#First, here's a reminder of our data format:

stats_df.head()
# Calculate correlations

corr = stats_df.corr()

 

# Heatmap

sns.heatmap(corr);
# Histograms allow you to plot the distributions of numeric variables.





# Distribution Plot (a.k.a. Histogram)

sns.distplot(df.Attack)

sns.distplot(df.Defense);
# Count Plot (a.k.a. Bar Plot)

sns.countplot(x='Type 1', data=df, palette=pkmn_type_colors)

 

# Rotate x-labels

plt.xticks(rotation=-45);
#Factor plots make it easy to separate plots by categorical classes.



# Factor Plot

g = sns.catplot(x='Type 1', 

                   y='Attack', 

                   data=df, 

                   hue='Stage',  # Color by stage

                   col='Stage',  # Separate by stage: comment out this value if you want all the data on a single plot

                   kind='swarm') # Swarmplot

 

# Rotate x-axis labels

g.set_xticklabels(rotation=-45);

 

# Doesn't work because only rotates last plot

# plt.xticks(rotation=-45)
#Density plots display the distribution between two variables.



#Tip: Consider overlaying this with a scatter plot.



# Density Plot

sns.kdeplot(df.Attack, df.Defense);
# Joint Distribution Plot

sns.jointplot(x='Attack', y='Defense', data=df);