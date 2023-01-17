# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
import pandas as pd
data = pd.read_csv("../input/deliveries.csv")

data.head(100)

matches =pd.read_csv("../input/matches.csv")

matches.head()
matches["winner"].value_counts().plot(kind= "bar")
matches["toss_decision"].value_counts(dropna=False)
matches.columns
matches.player_of_match.value_counts().head(10)
matches.venue.value_counts().head(10)
# Dharmasena is the top Umpire 

matches["umpire1"].value_counts().head(10)

                                 
# so only 3 matches with no result.

matches["result"].value_counts()
win_runs = matches["win_by_runs"] > 0  

win_runs.value_counts().plot(kind="bar")
win_wickets = matches["win_by_wickets"] > 0

win_wickets.value_counts()
data = pd.read_csv("../input/deliveries.csv")

data.head(10)
win_by_runs =  matches["win_by_runs"] > 0

win_by_runs.value_counts()
win_by_wik = matches["win_by_wickets"] > 0

win_by_wik.value_counts()
data.head()
data["extra_runs"].sum()
data["total_runs"].sum()
grouped = matches.groupby("season")

grouped.size()

        

matches.head()
grouped["winner"].size()
grouped = matches.groupby("season")["winner"]

winner_by_season = grouped.apply(lambda x: x.value_counts())

print(winner_by_season)

matches.head()
matches["win_by_runs"].plot(kind="hist", bins=5)
# Smooth version of Histogram

matches["win_by_runs"].plot(kind="density", xlim=(0,200))
pd.scatter_matrix(matches[["win_by_runs","win_by_wickets"]], figsize=(10,8))
matches.boxplot(column="win_by_runs", by="season", figsize=(8,5))
matches.boxplot(by="season", figsize=(8,5))
matches.hist(column="win_by_runs", by="season", figsize=(10,7))