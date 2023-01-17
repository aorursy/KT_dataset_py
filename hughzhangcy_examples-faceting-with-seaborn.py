import pandas as pd

import seaborn as sns



pokemon = pd.read_csv("../input/Pokemon.csv", index_col=0)

pokemon.head(3)
g = sns.FacetGrid(pokemon, row="Legendary")

g.map(sns.kdeplot, "Attack")
g = sns.FacetGrid(pokemon, row="Generation", col="Legendary")

g.map(sns.kdeplot, "Attack")
sns.pairplot(pokemon[['HP', 'Attack', 'Defense']])