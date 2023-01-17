import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/albumlist.csv", encoding = 'latin1') # I've added that 'encoding' bit just to fix the unicode decoding error in python 3

df.loc[0:20]
# Cleaning up genres

split_genre = []

for s in  df["Genre"]:

    split_genre.append(s.split(",")[0]) # Split every genre field entry at the comma

df["Genre"] = split_genre               # and only use the first genre specified

df.loc[0:20]
# Adding decades column

newyears = []

for year in df["Year"]:

    if year < 1960:

        newyears.append("50s")

    elif year < 1970:

        newyears.append("60s")

    elif year < 1980:

        newyears.append("70s")

    elif year < 1990:

        newyears.append("80s")

    elif year < 2000:

        newyears.append("90s")

    elif year < 2010:

        newyears.append("00s")

    else:

        newyears.append("10s")

df["Decade"] = newyears

sorter = ["50s", "60s", "70s", "80s", "90s", "00s", "10s"]

df["Decade"] = pd.Categorical(df["Decade"], sorter)

df = df.sort_values("Decade")

df.head()
group_artist = df.groupby("Artist")

ser = group_artist.size()
byartist = pd.DataFrame({"Artist":ser.index, "Count":ser.values})

topartists = byartist.sort_values("Count", ascending = False).iloc[0:20]

topartists.index = [x for x in range(1, 21)] # Reset the numbering to start from 1

topartists
genre_series = df.groupby("Genre").size()

bygenre = pd.DataFrame({"Genre":genre_series.index, "No. of Entries":genre_series.values})

topgenres = bygenre.sort_values("No. of Entries", ascending = False)

topgenres.index = [x for x in range(1, 12)]

topgenres
ser = df.groupby(["Decade", "Genre"]).size()

years = []

genres = []

for x in ser.index:

    years.append(x[0])

    genres.append(x[1])

byyear = pd.DataFrame({"Decade":years, "Genre":genres, "Count":ser.values})

# Order Decade chronologically

sorter = ["50s", "60s", "70s", "80s", "90s", "00s", "10s"]

byyear["Decade"] = pd.Categorical(byyear["Decade"], sorter)

byyear = byyear.sort_values("Decade")
import seaborn as sns

fig = plt.figure(figsize = (20, 20))

sns.set_style("whitegrid")

ax = sns.factorplot(x="Decade", y="Count", hue = "Genre", aspect = 2,

                    data=byyear, kind = "bar", size = 10, 

                    palette = sns.color_palette("hls", 8), legend = False)

plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), prop={'size':15})

plt.title("Most Popular Genres by Decade", fontsize = 20)

plt.xlabel("Decade", fontsize = 17)

plt.ylabel("Number of Entries", fontsize = 17)
rock = df[df.Genre == "Rock"]

# Now to split each subgenre into separate strings and add them to a list

def split_sub(sub):

    res = []

    for x in sub:

        spl = x.split(", ") # To separate the sub-genres using comma as the delimiter

        for s in spl:       # into 'spl', which is a list

            res.append(s)

    return res



subgenres = split_sub(list(rock["Subgenre"]))
import collections as co



count = co.Counter(subgenres)

topsub_list = count.most_common() # 'topsub_list' is now a list of tuples

topsub = pd.DataFrame(topsub_list, columns = ["Subgenre", "Number of entries"])

topsub.index = [x for x in range(1, len(topsub) + 1)]

topsub.loc[0:20]
art_ser = df.groupby(["Decade", "Artist"]).size()

temp = art_ser.to_frame()

temp = temp.rename(columns = {0:'No. of Entries'})

art_dec = temp['No. of Entries'].groupby(level = 0, group_keys = False)

art_dec.nlargest(5)