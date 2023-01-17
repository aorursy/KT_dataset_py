import numpy as np 

# Pandas is a good library for managing datasets

import pandas as pd 



# Matplotlib allows for additional customization

# %matplotlib inline to display our plots inside your notebook.

from matplotlib import pyplot as plt

%matplotlib inline



# Seaborn for plotting and styling

import seaborn as sns
combats = pd.read_csv("../input/pokemon/combats.csv")

pokemon = pd.read_csv("../input/pokemon/pokemon.csv")

tests = pd.read_csv("../input/pokemon/tests.csv")

# head function displays the first five rows. 

pokemon.head()
pokemon.columns
sns.lmplot(x="Attack", y="Defense", data=pokemon);

 

# An Alternative way

#sns.lmplot(x=df.Male_Pct, y=df.Female_Pct)
# Adding a bit more style and a legendary filter

sns.set_style('whitegrid')

sns.lmplot(

    x="Attack",

    y="Defense",

    data=pokemon,

    fit_reg=False,

    hue='Legendary',

    palette="Set1")
sns.set_style('darkgrid')  #changes the background of the plot

plt.figure(figsize=(14, 6))

sns.regplot(

    x="Attack", y="Defense", data=pokemon,

    fit_reg=True)  #fit_Reg fits a regression line
# We can make faceted plots where we can segment plots based on another categorical variable: Generation in this case



plt.figure(figsize=(20, 6))

sns.set_style('whitegrid')

sns.lmplot(

    x="Attack",

    y="Defense",

    data=pokemon,

    fit_reg=False,

    hue='Legendary',

    col="Generation",

    aspect=0.4,

    height=10)
# We can also see plot a continous variable against a categorical column. 

# Below we're trying to see relationship between Speed and Legendary status



plt.figure(figsize=(14, 6))

sns.set_style('whitegrid')

sns.regplot(x="Legendary", y="Speed", data=pokemon)
# One issue with this plot is we cannot see the distribution at each value of speed as the points are overlapping. 

# This can be fixed by an option called jitter



plt.figure(figsize=(14, 6))

sns.set_style("ticks")

sns.regplot(x="Legendary", y="Speed", data=pokemon, x_jitter=0.3)
plt.figure(figsize=(14, 6))

sns.set_style("ticks")

sns.regplot(x="Attack", y="Legendary", data=pokemon, logistic=True)
plt.figure(figsize=(12, 6))

ax = sns.distplot(

    pokemon['Defense'], kde=True,

    norm_hist=False)  #norm_hist normalizes the count

ax.set_title('Defense')

plt.show()
plt.figure(figsize=(12, 6))

sns.jointplot(x='Attack', y='Defense', data=pokemon)
plt.figure(figsize=(12, 6))

sns.jointplot(x='HP', y='Speed', data=pokemon, kind='kde')
# Kind = hex is interesting

plt.figure(figsize=(12, 6))

sns.jointplot(x='HP', y='Speed', data=pokemon, kind='hex')
sns.pairplot(

    pokemon,

    hue='Legendary',

    vars=['Speed', 'HP', 'Attack', 'Defense', 'Generation'],

    diag_kind='kde')
plt.figure(figsize=(20, 6))

ax = sns.countplot(x="Type 1", data=pokemon, color='green')
plt.figure(figsize=(20, 6))

sns.countplot(

    x="Type 1", data=pokemon, hue='Legendary',color='green',

    dodge=False)  #dodge = False option is used to make stacked plots
sns.set_style('darkgrid')

plt.figure(figsize=(20, 6))

sns.barplot(x="Type 1", y='Speed', data=pokemon, color='green')
sns.set_style('darkgrid')

plt.figure(figsize=(20, 6))

sns.barplot(x="Type 1", y='Speed', data=pokemon, hue='Legendary')
plt.figure(figsize=(20, 6))

sns.pointplot(x="Generation", y='Speed', data=pokemon, hue='Legendary')
plt.figure(figsize=(12, 6))

sns.stripplot(x="Generation", y="Speed", data=pokemon)
plt.figure(figsize=(12, 6))

sns.stripplot(x="Generation", y="Speed", data=pokemon, jitter=0.4)
sns.set_style('ticks')

plt.figure(figsize=(12, 6))

sns.swarmplot(x="Generation", y="Speed", data=pokemon, hue='Legendary')
sns.boxplot(data=pokemon)
# Pre-format

stats_pokemon = pokemon.drop(['Generation', 'Legendary'], axis=1)

sns.boxplot(data=stats_pokemon)
# Set theme

sns.set_style('whitegrid')

 

# Violin plot

plt.figure(figsize=(15, 6))

sns.violinplot(x='Type 1', y='Attack', data=pokemon)
grid1 = sns.FacetGrid(data=pokemon, col='Generation', col_wrap=3)



grid1.map(plt.hist, "Speed")
# Something a little more complex

grid2 = sns.FacetGrid(data=pokemon, col='Generation', col_wrap=3, hue="Legendary")



grid2.map(sns.regplot, "Speed", "HP", fit_reg=False).add_legend()
grid3 = sns.FacetGrid(

    data=pokemon, col='Generation', row='Legendary', margin_titles=True)



grid3.map(sns.regplot, "Speed", "HP", fit_reg=False)