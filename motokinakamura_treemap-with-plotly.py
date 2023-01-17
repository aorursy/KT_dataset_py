import pandas as pd

import numpy as np



import plotly.express as px

import plotly.graph_objects as go
df = pd.read_csv('../input/top50spotify2019/top50.csv',encoding = "ISO-8859-1")
#sample

fig = go.Figure(go.Treemap(

    labels = ["Eve","Cain", "Seth", "Enos", "Noam", "Abel", "Awan", "Enoch", "Azura"],

    parents = ["", "Eve", "Eve", "Seth", "Seth", "Eve", "Eve", "Awan", "Eve"]

))



fig.show()
#make a df it's grouped by "Genre"

gb_genre =df.groupby("Genre").sum()



gb_genre.head()
genre = list(gb_genre.index)

score = list(gb_genre.Popularity)



print(genre)

print(score)
#first treemap

test_tree = go.Figure(go.Treemap(

    labels =  genre,

    parents=[""]*len(genre),

    values =  score,

    textinfo = "label+value"

))



test_tree.show()
#first treemap

test_tree_blue = go.Figure(go.Treemap(

    labels =  genre,

    parents=[""]*len(genre),

    values =  score,

    textinfo = "label+value",

    marker_colorscale = 'Greens'

))



test_tree_blue.show()
gb_ArtistName = df.groupby("Artist.Name").sum()
artist = list(gb_ArtistName.index)

score_artist = list(gb_ArtistName.Popularity)
#first treemap

test_tree_artist = go.Figure(go.Treemap(

    labels =  artist,

    parents=[""]*len(artist),

    values =  score_artist,

    textinfo = "label+value"

))



test_tree_artist.show()
def test_func(str):

    if "pop" in str:

        return "POP"

    elif "hip hop" in str:

        return "HIP HOP"

    elif "rap" in str:

        return "RAP"

    else:

        return "OTHER"
test_func("awefawoefoawrapfewaofawoe")
test_df = df.Genre.map(lambda x : test_func(x))
df["big_genre"] = test_df
gb_big_genre =df.groupby("big_genre")
genre_2 = list(gb_big_genre.sum().index)

score_2 = list(gb_big_genre.sum().Popularity)
#second

test_tree_2 = go.Figure(go.Treemap(

    labels =  genre_2,

    parents=[""]*len(genre_2),

    values =  score_2,#labelsの要素に対応する値をいれる

    textinfo = "label+value"

))



test_tree_2.show()
genre_3 = list(gb_genre.index)

score_3 = list(gb_genre.Popularity)





print(genre,score, len(genre),len(score))
big_genre = list(gb_big_genre.sum().index)

big_score = list(gb_big_genre.sum().Popularity)



print(big_genre,big_score)
genre.extend(big_genre)

score.extend(big_score)



print(genre,score)
parent = list(gb_genre.index.map(lambda x : test_func(x)))



print(parent)

print("---")

print(len(parent))
big_parent = [""]*4

parent.extend(big_parent)



print(parent)
test_tree_3 = go.Figure(go.Treemap(

    labels =  genre,

    #parents = [""]*len(labels),

    parents=parent,

    values =  score,#labelsの要素に対応する値をいれる

    textinfo = "label+value"

))



test_tree_3.show()