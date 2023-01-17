import pandas as pd

import numpy as np
import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))
df1 = pd.read_csv("/kaggle/input/spotify-dataset-19212020-160k-tracks/data_w_genres.csv")
df1.head()
df2 = pd.read_csv("/kaggle/input/spotify-dataset-19212020-160k-tracks/data.csv")
df2.head()
# Making a new dataframe genre containing only artists and genre. Cleaning the dataframe df1.



genre=df1[["genres","artists"]]

genre=genre[genre["genres"]!="[]"]

genre["genres"]=genre["genres"].str.replace("'", "")

genre["genres"]=genre["genres"].str.replace("[", "")

genre["genres"]=genre["genres"].str.replace("]", "")
# Exploring Genre dataframe

genre.head(50)
# Cleaning the dataframe df2 that contains song details.

df2.head()

songs = df2

songs["artists"]=songs["artists"].str.replace("[", "")

songs["artists"]=songs["artists"].str.replace("]", "")

songs["artists"]=songs["artists"].str.replace("'", "")

songs.head()
merged=pd.merge(genre, songs)
merged.head()
rock_or_punk_merged = merged[merged["genres"].str.contains("rock|punk")]

rock_or_punk_merged.head()
rocknrun_150_165bpm = rock_or_punk_merged[(rock_or_punk_merged.tempo >= 150)&(rock_or_punk_merged.tempo <=165)]

rocknrun_150_165bpm.head()
top200 = rocknrun_150_165bpm.sort_values(['popularity'], ascending=False).head(200)

top200_rocknrun_150_165bpm = top200[["artists","name"]]
top200_rocknrun_150_165bpm.rename(columns={'artists':'ARTIST','name': 'TITLE'}).to_csv(r'top200_rocknrun_150_165bpm.csv')
rocknrun_165_170bpm = rock_or_punk_merged[(rock_or_punk_merged.tempo > 165)&(rock_or_punk_merged.tempo <=170)]

rocknrun_165_170bpm.head()
top200 = rocknrun_165_170bpm.sort_values(['popularity'], ascending=False).head(200)

top200_rocknrun_165_170bpm = top200[["artists","name"]]
top200_rocknrun_165_170bpm.rename(columns={'artists':'ARTIST','name': 'TITLE'}).to_csv(r'top200_rocknrun_165_170bpm.csv')
rocknrun_170_185bpm = rock_or_punk_merged[(rock_or_punk_merged.tempo > 170)&(rock_or_punk_merged.tempo <=185)]

rocknrun_170_185bpm.head()
top200 = rocknrun_170_185bpm.sort_values(['popularity'], ascending=False).head(200)

top200_rocknrun_170_185bpm = top200[["artists","name"]]
top200_rocknrun_170_185bpm.rename(columns={'artists':'ARTIST','name': 'TITLE'}).to_csv(r'top200_rocknrun_170_185bpm.csv')