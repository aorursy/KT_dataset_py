import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

#import libraries
pokemon = pd.read_csv('../input/Pokemon.csv')

#import data
# Looking at the breakdown of type 1

type_count = pd.value_counts(pokemon['Type 1'], sort = True).sort_index()

labels = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fariy', 'Fighting', 'Fire',

         'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock', 'Steel', 'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie1.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data', y = 1.05, size = 15)

sns.heatmap(pokemon.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
pokemon.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
df1 = pokemon[pokemon.Generation == 1]

#look at just first generation pokemon
# Looking at the breakdown of type 1

type_count = pd.value_counts(df1['Type 1'], sort = True).sort_index()

#print(type_count)

labels = ['Bug', 'Dragon', 'Electric', 'Fariy', 'Fighting', 'Fire',

          'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock',  'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie2.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data', y = 1.05, size = 15)

sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
df1.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
df1 = pokemon[pokemon.Generation == 2]

#look at gen 2 pokemon
# Looking at the breakdown of type 1

type_count = pd.value_counts(df1['Type 1'], sort = True).sort_index()

#print(type_count)

labels = ['Bug', 'Dark', 'Electric', 'Fariy', 'Fighting', 'Fire',

          'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock', 'Steel', 'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie3.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data', y = 1.05, size = 15)

sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
df1.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
df1 = pokemon[pokemon.Generation == 3]

#look at gen 3 pokemon
# Looking at the breakdown of type 1

type_count = pd.value_counts(df1['Type 1'], sort = True).sort_index()

#print(type_count)

labels = ['Bug', 'Dark',  'Electric', 'Fariy', 'Fighting', 'Fire',

          'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock', 'Steel', 'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie4.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data', y = 1.05, size = 15)

sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
df1.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
df1 = pokemon[pokemon.Generation == 4]

#look at gen 4 pokemon
# Looking at the breakdown of type 1

type_count = pd.value_counts(df1['Type 1'], sort = True).sort_index()

#print(type_count)

labels = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fariy', 'Fighting', 'Fire',

          'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock', 'Steel', 'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie5.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data', y = 1.05, size = 15)

sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
df1.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
df1 = pokemon[pokemon.Generation == 5]

#look at gen 5 pokemon
# Looking at the breakdown of type 1

type_count = pd.value_counts(df1['Type 1'], sort = True).sort_index()

#print(type_count)

labels = ['Bug', 'Dark', 'Dragon', 'Electric',  'Fighting', 'Fire',

         'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock', 'Steel', 'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie6.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data', y = 1.05, size = 15)

sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
df1.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
df1 = pokemon[pokemon.Generation == 6]

#look at gen 6 pokemon
# Looking at the breakdown of type 1

type_count = pd.value_counts(df1['Type 1'], sort = True).sort_index()

#print(type_count)

labels = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fariy', 'Fighting', 'Fire',

         'Flying', 'Ghost', 'Grass',  'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock', 'Steel', 'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie7.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data', y = 1.05, size = 15)

sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
df1.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
df1 = pokemon['Legendary'].map(lambda x: x == 0)

df1 = pokemon[df1]

#get the data without legendaries
# Looking at the breakdown of type 1

type_count = pd.value_counts(df1['Type 1'], sort = True).sort_index()

labels = ['Bug', 'Dark', 'Dragon', 'Electric', 'Fariy', 'Fighting', 'Fire',

         'Flying', 'Ghost', 'Grass', 'Ground', 'Ice', 'Normal', 'Poison', 'Psychic',

         'Rock', 'Steel', 'Water']

sizes = type_count



fig1, ax1 = plt.subplots()

ax1.pie(sizes, labels = labels, startangle=90)

ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.



plt.title('Number of Pokemon by type 1')

#plt.savefig('pie8.png', bbox_inches='tight')

plt.show()
colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data without legendaries', y = 1.05, size = 15)

sns.heatmap(df1.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations

colormap = plt.cm.plasma

plt.figure(figsize=(16,12))

plt.title('Pearson correlation of data with legendaries', y = 1.05, size = 15)

sns.heatmap(pokemon.corr(), linewidths=0.1, vmax=1.0, square=True, cmap=colormap, linecolor='black', annot=True)

plt.show()

#color map looking at Pearson correlations
df1.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution
pokemon.hist()

fig=plt.gcf()

fig.set_size_inches(20,15)

plt.show()

#histograms of the different stats to show if they have normal distribution