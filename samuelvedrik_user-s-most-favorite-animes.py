import numpy as np

import matplotlib.pyplot as plt

import pandas as pd

import json
df_animes = pd.read_csv("../input/myanimelist-dataset-animes-profiles-reviews/animes.csv")

df_profiles = pd.read_csv("../input/myanimelist-dataset-animes-profiles-reviews/profiles.csv")



df_animes = df_animes.drop_duplicates().reset_index(drop=True)

df_animes.set_index("uid", inplace=True)

df_profiles = df_profiles.drop_duplicates().reset_index(drop=True)
df_profiles["favorites_anime"] = df_profiles["favorites_anime"].str.replace("'", "")



def convert_to_list(lst):

    lst = lst.strip("[]")

    if lst == "":

        return []

    else:

        return list(map(int, lst.split(", ")))

    

df_profiles["favorites_anime"] = df_profiles["favorites_anime"].apply(convert_to_list)
favorites_count = {}



def count_favs(x):

    for uid in x:

        if uid not in favorites_count:

            favorites_count[uid] = 0

        favorites_count[uid] +=1 

        

_ = df_profiles["favorites_anime"].apply(count_favs)    
fav_counts = pd.DataFrame.from_dict(favorites_count, orient="index").rename(columns={0:"Count"})

df_fav = fav_counts.sort_values(by="Count", ascending=False).head(10)
df_fav["title"] = df_animes.loc[df_fav.index]["title"]
df_fav[["title", "Count"]]