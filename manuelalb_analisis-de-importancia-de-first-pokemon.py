# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))



# Any results you write to the current directory are saved as output.
df_battles= pd.read_csv("../input/battles.csv")

df_battles.head()
battles = [tuple(row) for row in (df_battles[["First_pokemon", "Second_pokemon"]].values)]

len(battles)
len(battles) - len(set(battles))
grouped = df_battles.groupby(["First_pokemon", "Second_pokemon"])

len(battles) - len(grouped.groups)
counters = grouped.count()

duplicates = list(counters[counters.Winner == 2].index)

triplicates = list(counters[counters.Winner == 3].index)

quadruplicates_or_more = list(counters[counters.Winner > 3].index)

len(duplicates), len(triplicates), len(quadruplicates_or_more)
def battles(p1, p2):

    return df_battles[(df_battles["First_pokemon"] == p1) & (df_battles["Second_pokemon"] == p2)]



duplicates_winners = [

    len(set(battles(p1, p2)["Winner"].values)) for p1, p2 in list(counters[counters.Winner > 1].index)

]



duplicates_winners[0:10]
set(duplicates_winners)
normal = [tuple(row) for row in (df_battles[["First_pokemon", "Second_pokemon"]].values)]

revers = [tuple(row) for row in (df_battles[["Second_pokemon", "First_pokemon"]].values)]

duplications = list(set(normal).intersection(set(revers)))

len(duplications)
def both_battles(p1, p2):

    b1 = df_battles[(df_battles.First_pokemon == p1) & (df_battles.Second_pokemon == p2)]

    b2 = df_battles[(df_battles.First_pokemon == p2) & (df_battles.Second_pokemon == p1)] 

    return b1.Winner.values[0] == b2.Winner.values[0]



missmatch = [(x, y) for (x, y) in duplications if both_battles(x,y)]

len(missmatch)
df_pokemon = pd.read_csv("../input/pokemon.csv")

f_pokemon = df_pokemon.rename(lambda x: "f_%s" % x, axis="columns")

s_pokemon = df_pokemon.rename(lambda x: "s_%s" % x, axis="columns")

missmatch_battles = pd.concat([battles(p1,p2) for p1,p2 in missmatch])

missmatch_battles = missmatch_battles.merge(f_pokemon, left_on="First_pokemon", right_on="f_#")

missmatch_battles = missmatch_battles.merge(s_pokemon, left_on="Second_pokemon", right_on="s_#")

missmatch_battles
220 / 3644 * 100