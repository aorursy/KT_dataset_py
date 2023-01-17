import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neighbors import NearestNeighbors



plt.rcParams["figure.figsize"] = (15, 7)

plt.style.use("ggplot")
FILE_PATH = "/kaggle/input/complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv"

df_pokemon_full = pd.read_csv(FILE_PATH)

df_pokemon_full = df_pokemon_full.drop("Unnamed: 0", axis=1)
INFO_CATEGORIES = ["pokedex_number", "name",  "type_1", "type_2", "ability_1", "ability_2", "ability_hidden", 

                   "status", "egg_type_1", "egg_type_2"]

STATS_CATEGORIES = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]



df_pokemon = df_pokemon_full[INFO_CATEGORIES + STATS_CATEGORIES].copy()
NN = NearestNeighbors(n_neighbors = 5, algorithm="ball_tree")

NN.fit(df_pokemon[STATS_CATEGORIES])



dist, ind = NN.kneighbors([df_pokemon[STATS_CATEGORIES].iloc[0]])
from IPython.display import display, HTML



pokemon = df_pokemon.sample(1)

pokemon_index = np.where(df_pokemon["name"] == pokemon["name"].iloc[0])[0][0]



dist, ind = NN.kneighbors(pokemon[STATS_CATEGORIES])

ind = list(ind.reshape(-1))

display(df_pokemon.iloc[[i for i in ind if i != pokemon_index]])



print("Hint: Pokemon abilities are:")

print(pokemon["ability_1"].iloc[0])

print(pokemon["ability_2"].iloc[0]) if pokemon["ability_2"].iloc[0] is not np.nan else None

print(pokemon["ability_hidden"].iloc[0]) if pokemon["ability_hidden"].iloc[0] is not np.nan else None



print("Distances:")

print(list(dist.reshape(-1)[1:]))
pokemon