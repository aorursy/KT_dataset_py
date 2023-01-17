import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

from hvplot import hvPlot

import hvplot.pandas

import holoviews as hv

from datetime import datetime



%matplotlib inline
imdb=pd.read_csv('imdb_details.csv')

imdb
imdb.dtypes
(imdb

.groupby("title")

 [["rating", "season"]]

.median()

.sort_values(by="rating", ascending=True)

.head(10)

)



##The worst episode of Dr Who were "The Tsurange Conundrum", "Arachnids in the UK", "The Battle of Ranskorr Av Kolor" from 11 season
(imdb

.groupby("title")

 [["rating", "season"]]

.median()

.sort_values(by="rating", ascending=False)

.head(19)

)



## The best episodes of Dr.Who are "Blink", "Heaven Send"
## Ratings and The Seasons

p=(imdb

.groupby("season")

[["rating"]]

.mean()

.sort_values(by="rating", ascending=False)

)

p



##The best ratings were for the seasons 4 and 9

## The worst ratings were for the Season 10 and 11
p.hvplot.bar( x="season", y="rating", title="With the next season, the ratings were slightly getting down")
# Number of Voters & Seasons 



(imdb

.groupby("season")

 [["nbr_votes"]]

.mean()

.sort_values(by="nbr_votes", ascending=True)

).plot.bar(y="nbr_votes")

imdb.corr() ## No positive correlation between fields
dw=pd.read_csv('dwguide.csv')



dw["summary"]=dw["summary"].fillna("None")



dw['broadcastdate']=pd.to_datetime(dw["broadcastdate"],format="%d %b %Y")

dw["broadcasthour"]=pd.to_datetime(dw['broadcasthour'], format="%I:%Mpm").dt.time



dw["views"]=dw["views"].str.replace(".", "").str.replace("m", "000000").astype(np.int64)



dw["Doctor"]=dw.cast.str[30:43]



dw=dw.drop(["cast"], axis=1)

dw
dw["broadcastdate"].dtypes
def missing_values(n):

    df=pd.DataFrame()

    df["missing, %"]=dw.isnull().sum()*100/len(dw.isnull())

    df["missing, num"]=dw.isnull().sum()

    return df.sort_values(by="missing, %", ascending=False)

missing_values(dw)
dw
dw["Doctor"].value_counts()
dwguide=pd.read_csv("dwguide.csv")

dwguide['broadcastdate']=pd.to_datetime(dwguide["broadcastdate"],format="%d %b %Y")



dwguide.head(5)
dwguide.sort_values(by="broadcastdate", ascending=True).head(5)
dwguide.sort_values(by="broadcastdate", ascending=False).head(5)