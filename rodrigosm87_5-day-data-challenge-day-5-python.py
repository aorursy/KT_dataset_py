import pandas as pd

import scipy.stats

import matplotlib.pyplot as plt



pokemon = pd.read_csv("../input/Pokemon.csv")

pokemon.head()
pokemonType1 = pokemon["Type 1"].value_counts()

scipy.stats.chisquare(pokemonType1)
pokemonType2 = pokemon["Type 2"].value_counts()

scipy.stats.chisquare(pokemonType2)
pokemonTypes = pokemonType1.to_frame()

pokemonTypes["Type 2"] = pokemonType2
pokemonTypes.transpose()
fig, ax = plt.subplots(1, 1, figsize=(14, 8))

pokemonTypes.plot.bar(ax=ax,)
contingencyTable = pd.crosstab(pokemon["Type 1"], pokemon["Type 2"])

contingencyTable
scipy.stats.chi2_contingency(contingencyTable)