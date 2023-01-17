import numpy as np

import pandas as pd

import networkx as nx

import matplotlib.pyplot as plt

import seaborn as sns
df =  pd.read_csv('../input/pokemon.csv')

df.head(n=10)                              #print the first 10 rows of the table
df.info()
df.type1.value_counts().plot(kind='pie', title="Pokemon per type", autopct="%1.1f%%", figsize=(12,12), legend=False)
columns = [

    "type1",

    "weight_kg",

    "height_m",

    "percentage_male"

]



attr_df = df.loc[:, columns]

cm = attr_df["height_m"].apply(lambda x: x * 100)

attr_df["height_cm"] = cm

attr_df = attr_df.drop(["height_m"], axis=1)

mean_grouped_type = attr_df.groupby(["type1"]).mean()

mean_grouped_type.plot(kind="bar", figsize=(12, 12))
df_stats = df.loc[:, ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "base_total"]]

df_stats.describe()
df_stats.plot(subplots=True, figsize=(12, 14))
df.loc[:,["base_total", "is_legendary"]].plot(subplots=True, figsize=(12, 6))
p_df = df.loc[:, ["name", "type1", "type2", "is_legendary", "pokedex_number", "hp", "attack", "defense", "sp_attack", "sp_defense", "speed", "base_total", "height_m", "weight_kg"]]

p_df.nlargest(15, "base_total")
plt.subplots(figsize = (15,5))

plt.title("Attack by Type1")

sns.boxplot(x="type1", y="attack", data=df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (15,5))

plt.title("Defense by Type1")

sns.boxplot(x="type1", y="defense", data=df)

plt.ylim(0,200)

plt.show()
plt.subplots(figsize = (15,5))

plt.title("Attack spe. by Type1")

sns.boxplot(x="type1", y="sp_attack", data=df)

plt.ylim(0,200)

plt.show()
p_types = [

  ["fire", "#EE8130"],

  ["water", "#6390F0"]

]



ax = None

for p_type in p_types:

    sub_df = df.loc[df["type1"] == p_type[0]]

    ax = sub_df.plot(x="defense", y="attack", figsize=(12,12), s=70, kind="scatter", ax=ax, color=p_type[1], alpha=0.5)
plt.figure(figsize=(10,6))

sns.heatmap(p_df.corr(), annot=True) #df.corr() makes a correlation matrix and sns.heatmap is used to show the correlations heatmap

plt.show()
# ident used to find pokemons

ident = "name" #"pokedex_number" # "name"



def calc_score(pok1, pok2):

    return 1.0
match_percent = 0.15

sample_size = 151

graph = []



# fill empty weight and height

df["weight_kg"].fillna(1, inplace = True)

df["height_m"].fillna(1, inplace = True)



sample = df.head(sample_size)



for index, row in sample.iterrows():

    matchs = []

    weight_diff = row.weight_kg * match_percent

    height_diff = row.height_m * match_percent

    ## exclude automaticaly pokemons with 'match_percent' difference in height 

    sub_df = sample.loc[(row.height_m - height_diff < df.height_m) & (df.height_m < row.height_m + height_diff)]

    for sub_index, sub_row in sub_df.iterrows():

        score = calc_score(row[ident], sub_row[ident])

        if score > 0.60 and row[ident] != sub_row[ident]:

            matchs.append((sub_row[ident], score))

    graph.append((row[ident], matchs))



print (graph[0])
G = nx.Graph()



for pok in graph:

    G.add_node(pok[0]) # ident

    for match in pok[1]:

        G.add_edge(pok[0], match[0])



plt.figure(figsize=(15, 15))

nx.draw(G, with_labels=True, pos=nx.circular_layout(G), node_size=50, font_size=10, width=0.5, alpha=0.3)

plt.show()