import pandas as pd

import seaborn as sns

#import numpy as np



pokemon = pd.read_csv("../input/Pokemon.csv", index_col=0)

pokemon.head(5)
df = pokemon[pokemon['Legendary'].isin([False, True])]

#df.head()

g = sns.FacetGrid(df, col="Legendary")

g.map(sns.kdeplot, "Attack")



g = sns.FacetGrid(pokemon, row="Legendary")

g.map(sns.kdeplot, "Attack")
g = sns.FacetGrid(pokemon, row='Generation', col='Legendary')

g.map(sns.kdeplot, 'Attack')
sns.pairplot(pokemon[['Defense', 'Attack', 'HP']])