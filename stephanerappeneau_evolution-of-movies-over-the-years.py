### Import des librairies utiles

import csv

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



# Chargement du fichier dans un dataframe / On limite à 1000 pour developper

df = pd.read_csv("../input/AllMoviesDetailsCleaned.csv", encoding='utf-8-sig', sep=";", 

                 engine="python",parse_dates=["release_date"])



# On remplit les vides avec 0 & on force le runtime en integer

df = df.fillna("0")

df['runtime'] = df['runtime'].astype(int)
# On crée une série avec l'année extraite (utilisation des fonctions map & lambda)

serie = df["release_date"].map(lambda x:x.year)



# tout d'abord nb de films par ans

nb_films_an = df["id"].groupby(pd.Index(serie)).count()



# On affiche la serie

plt.bar(nb_films_an.index,nb_films_an)

plt.title("number of movies by release year")

plt.show()
# puis la moyenne des durées des films >30 min et <300 min

duree_min = df["runtime"]>30

duree_max = df["runtime"]<300



# On crée une série avec l'année extraite (utilisation des fonctions map & lambda)

df2 = df[duree_min & duree_max].loc[:,["release_date","runtime"]]

serie = df2["release_date"].map(lambda x:x.year)



film_runtime = df2["runtime"].groupby(pd.Index(serie)).mean()



# On affiche la serie (on vire 1970 qui est l'année avec les dates par défaut, puis on ne garde que 1900=>2016)

film_runtime = film_runtime.loc[1900:2016].drop(1970)

plt.bar(film_runtime.index,film_runtime)

plt.title("evolution of average runtime over the years")

plt.show()
# most represented languages

test = df.groupby("original_language").count()

test.sort_values("id", ascending = False).head()
# On crée une série avec l'année extraite (utilisation des fonctions map & lambda)

serie = df["release_date"].map(lambda x:x.year)



# cumul du nombre de votes

cumul_vote = df["vote_count"].groupby(pd.Index(serie)).sum()



# On affiche la serie

plt.bar(cumul_vote.index,cumul_vote)

plt.title("number of votes per year")

plt.show()



# Conclusion : les gens se prononcent très peu sur les vieux films (ils ne les regardent probablement pas)
genre_film="Western"

serie2 = df["genres"].str.contains(genre_film)

df2 = df[serie2]

index_year1 = df["release_date"].map(lambda x:x.year)

index_year2 = df2["release_date"].map(lambda x:x.year)

df2.reindex(index_year2)

nb_genre_an = df2.groupby(pd.Index(index_year2)).count()

total_films_an = df["id"].groupby(pd.Index(index_year1)).count()



# On calcule la part relative des films de ce genre

for year in nb_genre_an.index:

    nb_genre_an.loc[year,"id"] = nb_genre_an.loc[year,"id"]*100/total_films_an.loc[year]



# On affiche la serie (on vire 1970 qui est l'année avec les dates par défaut, puis on ne garde que 1900=>2016)

nb_genre_an = nb_genre_an.loc[1900:2016].drop(1970)

plt.bar(nb_genre_an.index,nb_genre_an["id"])

plt.title("number of " + genre_film + " by year in %")

plt.show()