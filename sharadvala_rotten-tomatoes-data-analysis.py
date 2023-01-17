import matplotlib.pyplot as plt #for visualisaton

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns

%matplotlib inline
#Rotten tomatoes

rott_tomt = pd.read_csv("../input/rotten-tomato-movie-reviwe/rotten tomato movie reviwe.csv")



#getting overview of various columns

rott_tomt.info()
rott_tomt.head()
rott_tomt.plot.bar()
#finding dimensions

print(rott_tomt.shape)
#Let's see how many columns contains NA values

rott_tomt.isna().any()
rott_tomt.describe()
rott_tomt['Runtime'].hist(bins=50)
rott_tomt['TOMATOMETER Count'].hist(bins=30)
top_scored = rott_tomt.sort_values(["TOMATOMETER score","AUDIENCE score"], ascending=False)[

    ["Name", "Directed By", "TOMATOMETER score","AUDIENCE score"]]

top_scored.index = range(0,2100)

top_scored.head(n=15)
top_scored_count = rott_tomt.sort_values(["TOMATOMETER score","AUDIENCE score"], ascending=False)[

    ["Name", "Directed By", "TOMATOMETER score","AUDIENCE score", "AUDIENCE count"]]

top_scored_count.index = range(0,2100)

top_scored_count.head(n=15)
top_scored_runtime = rott_tomt.sort_values(["TOMATOMETER score", "AUDIENCE score"], ascending=False)[

    ["Name", "Directed By", "Runtime", "TOMATOMETER score","AUDIENCE score"]]

top_scored_runtime.index = range(0,2100)

top_scored_runtime.head(n=15)
#Let's plot with respect to Metascore because, it is more unique

top_scored_runtime.plot(kind="scatter",

                      x="Runtime",

                      y="AUDIENCE score",

                      alpha=0.4)
rott_tomt[["TOMATOMETER score", "AUDIENCE count"]].corr()
rott_tomt.plot(kind="scatter",

                      x="TOMATOMETER score",

                      y="AUDIENCE count",

                      color="red",

                      alpha=0.4,

                      )