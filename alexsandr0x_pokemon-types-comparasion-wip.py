import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

type_1 = "Dragon"
type_2 = "Dark"

sns.set(style="whitegrid", palette="muted")
pokemons = pd.read_csv("../input/Pokemon.csv")
pokemons = pokemons[~pokemons["Type 2"].notnull()]
pokemons = pokemons[((pokemons["Type 1"] == type_1) | (pokemons["Type 1"] == type_2))]
pokemons = pokemons[["Type 1", "Defense", "Attack", "Sp. Def", "Sp. Atk", "Speed"]]
pokemons = pd.melt(pokemons, "Type 1", var_name="stats")

sns.set(rc={'figure.figsize':(18.7,12.27)})
sns.set_context("paper", rc={"font.size":16,"axes.titlesize":18,"axes.labelsize":18}) 

ax = sns.violinplot(x="stats", y="value", hue="Type 1", data=pokemons, palette=["orange", "purple"], split=True)

ax.set_title("Comparation between two pokemon types", fontsize=32)
plt.setp(ax.get_legend().get_texts(), fontsize='18')
plt.setp(ax.get_legend().get_title(), fontsize='20')
pokemons = pd.read_csv("../input/Pokemon.csv")
pokemons = pokemons[(pokemons["Type 1"] == "Dragon") | (pokemons["Type 1"] == "Dark")]
pokemons = pokemons[["Type 1", "Defense", "Attack", "Sp. Def", "Sp. Atk", "Speed"]]
sns.pairplot(pokemons, hue="Type 1", palette=["orange", "purple"])