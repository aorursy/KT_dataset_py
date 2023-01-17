import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt

print(os.listdir("../input"))
df = pd.read_csv("../input/ig-public-lb/publicleaderboarddata/instant-gratification-publicleaderboard.csv")

df.sort_values(by = 'Score', ascending= False).head()



df = df[~((0.973689 < df["Score"] ) & ( df["Score"] < 0.973699 ))]

#df = df[~((0.973574 < df["Score"] ) & ( df["Score"] < 0.973589 ))]
df.tail()
P = df[["TeamId","TeamName","Score"]].groupby("TeamId").max()#.sort_values(by = 'Score', ascending= False)
Sorted = P.sort_values(ascending= False, by = "Score").head(120).reset_index()
Sorted.head(60)
Sorted.tail(60)