import numpy as np

import pandas as pd

from pandas import Series, DataFrame

import seaborn as sns

import matplotlib.pyplot as plt



sns.set_style('whitegrid')

%matplotlib inline
#loading the dataset as pokemon_df

pokemon_df = pd.read_csv('../input/pokemonGO.csv')



#taking a look at the first 10 rows

pokemon_df.head(n=10)
sns.factorplot('Type 2',data=pokemon_df,kind='count',size=7)

sns.factorplot('Type 1',data=pokemon_df,kind='count',size=(8))
sns.factorplot('Type 1', data=pokemon_df, kind='count', hue='Type 2', size=10)
pokemon_df[['Type 1','Type 2']].count()
#Question 1

pokemon_df[pokemon_df['Max CP'] == pokemon_df['Max CP'].max()]
#Question 2

pokemon_df[pokemon_df['Max CP'] == pokemon_df['Max CP'].min()]
#Question 3

pokemon_df[pokemon_df['Max HP'] == pokemon_df['Max HP'].max()]
#Question 4

pokemon_df[pokemon_df['Max HP'] == pokemon_df['Max HP'].min()]
#First off, what is the mean HP value?

mean_hp = pokemon_df['Max HP'].mean()

mean_hp
first_type_ground_df = pokemon_df[pokemon_df['Type 1'] == 'Ground']



second_type_ground_df = pokemon_df[pokemon_df['Type 2'] == 'Ground']
first_type_ground_df.count()
second_type_ground_df.count()
#the dataframe aren't don't have so many rows, so we can take a complete look at their content

first_type_ground_df
second_type_ground_df
#first dataframe

first_type_ground_df.plot(x='Name',y='Max HP',marker='o',figsize=(12,6),linestyle='--')

#setting the title of the dataframe

plt.title('Max HP pokemons whose first type is Ground')

#setting the y axis label

plt.ylabel('Max HP')

plt.axhline(mean_hp,color='red',linewidth=2)



second_type_ground_df.plot(x='Name', y='Max HP', marker='o',figsize=(12,5),linestyle='--')

#setting the title of the dataframe

plt.title('Max HP for pokemons whose second type is Ground')

#setting the y axis label

plt.ylabel('Max HP')

plt.axhline(mean_hp,color='red')
#finding the mean CP

mean_cp = pokemon_df['Max CP'].mean()

mean_cp
#creating a dataframe containing the strongest pokemons

strong_pokemon_df = pokemon_df[pokemon_df['Max CP'] >= mean_cp]

strong_pokemon_df = strong_pokemon_df[strong_pokemon_df['Max HP'] <= mean_hp]
strong_pokemon_df
strong_pokemon_df.count()
mean_hp2 = strong_pokemon_df['Max HP'].mean()

mean_cp2 = strong_pokemon_df['Max CP'].mean()
strong_pokemon_df = strong_pokemon_df[strong_pokemon_df['Max CP'] >= mean_cp2]

strong_pokemon_df = strong_pokemon_df[strong_pokemon_df['Max HP'] <= mean_hp2]
#Let's take a look at the new dataframe

strong_pokemon_df