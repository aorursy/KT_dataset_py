# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
game_highs=pd.read_csv("/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/game_highs_stats.csv")

game_highs.head()
biggerthan34=game_highs[game_highs["Age"]>34]

biggerthan34.sort_values("PTS",ascending=False)
biggerthan30=game_highs[game_highs["Age"]>30]

biggerthan30.sort_values("BLK",ascending=False)
import seaborn as sns

import matplotlib.pyplot as plt 

blk30=biggerthan30[biggerthan30["BLK"]>=3]

sns.countplot(data=blk30,x="Player")
totalstat=pd.read_csv("/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/totals_stats.csv")

totalstat.head()
prime_stats=totalstat[(totalstat["Age"]>26)& (totalstat["Age"]<=31)]

prime_stats.sort_values("STL",ascending=False)
bigger2000=prime_stats[prime_stats["PTS"]>=2000]

sns.countplot(data=bigger2000,x="Player")
bigger2000all=totalstat[totalstat["PTS"]>=2000]

sns.countplot(data=bigger2000all,x="Player")
pergame=pd.read_csv("/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/per_game_stats.csv")

pergame.head()
pergamepts30=pergame[(pergame["PTS"]>=30.0)&(pergame["RSorPO"]=="Regular Season")]

pergamepts30["Player"].value_counts()
sns.countplot(data=pergamepts30,x="Player")
pergamepts30=pergame[(pergame["AST"]>=7.0)&(pergame["RSorPO"]=="Regular Season")]

pergamepts30["Player"].value_counts()
sns.countplot(data=pergamepts30,x="Player")
pergamepts30=pergame[(pergame["PTS"]>=25.0)&(pergame["RSorPO"]=="Regular Season")]

pergamepts30["Player"].value_counts()
sns.countplot(data=pergamepts30,x="Player")
allgames=pd.read_csv("/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/allgames_stats.csv")

allgames.head()
p25as5=allgames[(allgames["PTS"]>=25)&(allgames["AST"]>5)]

sns.countplot(data=p25as5,x="Player")
p25as5=allgames[(allgames["PTS"]>=30)&(allgames["AST"]>5)]

sns.countplot(data=p25as5,x="Player")
p25as5=allgames[allgames["PTS"]>=50]

sns.countplot(data=p25as5,x="Player")
p25as5=allgames[allgames["TRB"]>10]

sns.countplot(data=p25as5,x="Player")
p25as5=allgames[allgames["STL"]>4]

sns.countplot(data=p25as5,x="Player")
p25as5=allgames[allgames["BLK"]>2]

sns.countplot(data=p25as5,x="Player")
p25as5=allgames[allgames["TOV"]>2]

sns.countplot(data=p25as5,x="Player")
advanced=pd.read_csv("/kaggle/input/michael-jordan-kobe-bryant-and-lebron-james-stats/advanced_stats.csv")

advanced.head()
advae=advanced[["Player","DBPM"]]

advae.sort_values("DBPM",ascending=False)[0:25]
advae.groupby("Player").sum()
advae.groupby("Player").mean()
advae=advanced[["Player","WS","RSorPO"]]

advae=advae[advae["RSorPO"]=="Regular Season"]

advae.groupby("Player").sum()
advae=advanced[["Player","PER","RSorPO"]]

advae=advae[advae["RSorPO"]=="Regular Season"]

advae.groupby("Player").mean()
advae=advanced[["Player","DWS","RSorPO"]]

advae=advae[advae["RSorPO"]=="Regular Season"]

advae.groupby("Player").mean()
advae=advanced[["Player","VORP","RSorPO"]]

advae=advae[advae["RSorPO"]=="Regular Season"]

advae.sort_values("VORP",ascending=False)
