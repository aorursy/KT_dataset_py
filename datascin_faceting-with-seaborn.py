import pandas as pd
import seaborn as sns

pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head(3)
g = sns.FacetGrid(pokemon, row="Legendary")
g.map(sns.kdeplot, "Attack")
g = sns.FacetGrid(pokemon, col="Legendary", row="Generation")
g.map(sns.kdeplot, "Attack")
sns.pairplot(pokemon[['HP', 'Attack', 'Defense']])