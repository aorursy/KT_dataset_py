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