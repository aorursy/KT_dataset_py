import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



from sklearn.neighbors import NearestNeighbors

from sklearn.decomposition import PCA, KernelPCA

from sklearn.cluster import KMeans



plt.rcParams["figure.figsize"] = (15, 7)

plt.style.use("ggplot")
FILE_PATH = "/kaggle/input/complete-pokemon-dataset-updated-090420/pokedex_(Update_05.20).csv"

df_pokemon = pd.read_csv(FILE_PATH)

df_pokemon = df_pokemon.drop("Unnamed: 0", axis=1)

df_pokemon.head(1)
INFO_CATEGORIES = ["pokedex_number", "name", "generation", "status", "type_1", "type_2"]

STATS_CATEGORIES = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]

df_pokemon = df_pokemon[INFO_CATEGORIES + STATS_CATEGORIES]

df_pokemon.head()
type_1_count = df_pokemon["type_1"].value_counts().sort_index() 

type_2_count = df_pokemon["type_2"].value_counts().sort_index()

type_count = (type_1_count + type_2_count).sort_values()



type_count.plot.barh()

_ = plt.title("Distribution of types"), plt.xlabel("Count")
def get_reflection(X):

    reflected = np.zeros(shape=(18, 18))



    for i in range(18):

        for j in range(18):

            reflected[j,i] = X[i,j]

    return reflected

    
type_1 = df_pokemon["type_1"]

type_2= df_pokemon[["type_1", "type_2"]].apply(

    lambda x : x.iloc[1] if not x.iloc[1] is np.nan else x.iloc[0],

    axis=1)

type_2.name = "type_2"



df_combo_count = pd.crosstab(type_1, type_2)
lower_triangle = np.tril(df_combo_count.to_numpy(), k=-1)

upper_triangle = np.triu(df_combo_count.to_numpy(), k=1)

diagonal = np.diag(np.diag(df_combo_count.to_numpy()))



upper_reflected = get_reflection(upper_triangle)



total_count = upper_reflected + lower_triangle

total_count = total_count + get_reflection(total_count)

total_count = total_count + diagonal
plt.imshow(total_count, cmap="OrRd")

plt.xticks(range(18), labels=list(df_combo_count.index))

plt.yticks(range(18), labels=list(df_combo_count.index))

_ = plt.xticks(rotation=90), plt.grid(b=None), plt.title("Type Combination HeatMap")
df_pokemon[STATS_CATEGORIES].describe()
from IPython.display import Image, display, HTML

import ipywidgets as widgets

from ipywidgets import interact, interact_manual

import plotly.graph_objects as go





pca = PCA(n_components=2)

pca.fit(df_pokemon[STATS_CATEGORIES])
import plotly.graph_objects as go



def plot_by_type_interactive(Type, Stat):

    

    if Type != "All":

        filter_ = (df_pokemon["type_1"] == Type) | (df_pokemon["type_2"] == Type)

        df = df_pokemon.loc[filter_]

    else:

        df = df_pokemon



    X = pca.transform(df[STATS_CATEGORIES])

    

    fig = go.Figure(go.Scatter(x=X[:, 0], 

                               y=X[:, 1],

                              mode="markers",

                              text=df["name"],

                              marker={"color": df[Stat], "showscale": True, "colorscale": "solar"}))



    fig.update_layout(title=f"Pokemon Stats PCA Visualization {Type} Type, by {Stat}")

    fig.show()



_ = interact(plot_by_type_interactive, Type = ["All"] + list(df_pokemon["type_1"].unique()), Stat = STATS_CATEGORIES)
def pokedex(Pokemon):

    

    poke_list = Pokemon.lower().split(", ")

    display(df_pokemon[df_pokemon["name"].apply(lambda x : x.lower() in poke_list)])

    

    

_ = interact_manual(pokedex, Pokemon = widgets.Text(value="Bulbasaur"))
fig = go.Figure(go.Scatter(x=df_pokemon["sp_defense"] + df_pokemon["defense"], 

                          y=df_pokemon["speed"], 

                          text=df_pokemon["name"], mode="markers"))

fig.update_layout(xaxis_title="defense + sp_defense", yaxis_title="speed", title="Correlation between speed and defense")

fig.show()
from sklearn.metrics import silhouette_score







sil_scores = []

inertias = []

for clusters in range(2, 21):

    clusterer = KMeans(n_clusters=clusters, random_state=42)

    labels = clusterer.fit_predict(df_pokemon[STATS_CATEGORIES])

    sil_scores.append(silhouette_score(df_pokemon[STATS_CATEGORIES], labels))

    inertias.append(clusterer.inertia_)


fig, ax = plt.subplots(1, 2)

ax[0].plot(range(2, 21), sil_scores)

ax[1].plot(range(2, 21), inertias)



ax[0].set_xlim([2, 20])

ax[1].set_xlim([2, 20])

ax[0].set_title("Silhouette Scores")

_ = ax[1].set_title("Inertia")
