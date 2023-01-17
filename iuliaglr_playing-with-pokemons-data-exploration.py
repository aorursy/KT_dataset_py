import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # 2D plotting library
import seaborn as sns # visualization library

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

pokemon_data = pd.read_csv('../input/pokemon.csv')
combats_data = pd.read_csv('../input/combats.csv')
tests_data = pd.read_csv('../input/tests.csv')
prediction_data = tests_data.copy()

# Quick look at the actual data values
pokemon_data.head(10)
# Overview of pokemon CSV - understanding the types of the attributes
pokemon_data.info()
# Different oveview of pokemon CSV - some statistics on the data values
pokemon_data.describe()
# Visualization of pokemon types using the 'Type1' feature

sns.countplot(x = 'Type 1', data = pokemon_data, order = pokemon_data['Type 1'].value_counts().index)
plt.xticks(rotation = 90) # make values on X vertical, rather than horizontal
plt.title ('Pokemon Types I')
plt.show()
# Visualization of pokemon types using the 'Type2' feature

sns.countplot(x = 'Type 2', data = pokemon_data, order = pokemon_data['Type 2'].value_counts().index)
plt.xticks(rotation = 90)
plt.title ('Pokemon Types II')
plt.show()
# Legendary vs Non-Legendary split
leg_split = [pokemon_data['Legendary'].value_counts()]
plt.pie(leg_split[0], autopct='%1.1f%%', colors = ['lightskyblue', 'orange'], explode = (0, 0.1), startangle=90)
plt.title('Legendary vs Non-Legendary Split', size = 14)
plt.legend(labels = ['Non-Legendary', 'Legendary'], loc = 'best')
plt.show()
# Legendary distribution by Generation
sns.countplot(x = 'Generation', hue = 'Legendary', data = pokemon_data)
plt.show()
# Correlation map
f, ax = plt.subplots(figsize = (18,18)) # set up the matplotlib figure
cmap = sns.diverging_palette(220, 10, as_cmap = True) # generate custom colormap
sns.heatmap(pokemon_data.corr(), annot = True, cmap = cmap, linewidths = 1, fmt = '.1f', ax = ax) # draw the map
# Line Plot for Speed & Defense
# alpha = opacity
pokemon_data.Speed.plot(kind = 'line', color = 'b', label = 'Speed', linewidth=1, alpha = 0.5, grid = True,linestyle = ':')
pokemon_data.Defense.plot(color = 'r', label = 'Defense', linewidth = 1, alpha = 0.7, grid = True, linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('x axis')              # label = name of label
plt.ylabel('y axis')
plt.title('Pokemon Speed and Defense')            # title = title of plot
# Scatter Plot for Attack & Defense
# x = attack, y = defense
pokemon_data.plot(kind='scatter', x='Attack', y='Defense', alpha = 0.5, color = 'red')
plt.xlabel('Attack')              # label = name of label
plt.ylabel('Defense')
plt.title('Attack Defense Scatter Plot')            # title = title of plot
# Histogram for Speed & Defense
pokemon_data.Speed.plot(kind = 'hist', bins = 50, figsize = (15,15))
pokemon_data.Defense.plot(kind = 'hist', bins = 50, alpha = 0.5, figsize = (15,15))
plt.title('Distribution of Speed and Defense')
# Plotting Attack, Defense, Speed
data1 = pokemon_data.loc[:,["Attack","Defense","Speed"]]
data1.plot()
# Plotting each data series separately
data1.plot(subplots = True)
# Calculation of Stats_Sum
pokemon_data['Stats_Sum'] = pokemon_data['HP'] + pokemon_data['Attack'] + pokemon_data['Defense'] + pokemon_data['Sp. Atk'] + pokemon_data['Sp. Def'] + pokemon_data['Speed']
print(pokemon_data.head(10))
# Stats_Sum plot
sns.distplot(pokemon_data.Stats_Sum, color = 'g')
plt.title('Pokemon Stats Distribution')
plt.show()
# Looking for top 3 weakest and top 3 strongest Pokemon in terms of sum of stats
ordered_pokemon_data = pokemon_data.sort_values(by = 'Stats_Sum')
print ('Weakest 3 Pokemon: ')
print(ordered_pokemon_data[['Name', 'Stats_Sum']].head(3))
print('-----------------------------------------')
print ('Strongest 3 Pokemon: ')
print(ordered_pokemon_data[['Name', 'Stats_Sum']].tail(3))
# Box plots for total stat of each generation
sns.set()
dim = (11, 8)
fig, ax = plt.subplots(figsize = dim)
sns.boxplot(x = 'Generation', y = 'Stats_Sum', data = pokemon_data, ax = ax)
plt.title ('Box Plot of Total Stats for each Generation')
# Checking the outlier seen in the box plot above
gen1 = pokemon_data[pokemon_data.Generation == 1]
gen1['Stats_Sum'].max()