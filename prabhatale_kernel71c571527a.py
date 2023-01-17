# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

import seaborn as sbn



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
data = pd.read_csv("/kaggle/input/international-football-results-from-1872-to-2017/results.csv")

data.info()
def replacing(dataset):

    dataset.loc[dataset["home_team"] == "Northern Ireland" , "home_team"] = "Ireland"

    dataset.loc[dataset["away_team"] == "Northern Ireland" , "away_team"] = "Ireland"

    dataset.loc[dataset["home_team"] == "Republic of Ireland" , "home_team"] = "Ireland"

    dataset.loc[dataset["away_team"] == "Republic of Ireland" , "away_team"] = "Ireland"

    dataset.loc[dataset["country"] == "Republic of Ireland" , "country"] = "Ireland"

    dataset.loc[dataset["country"] == "Northern Ireland" , "country"] = "Ireland"

    return dataset
def extract(dataframe, startyear,endyear):

    dataframe["date"] = pd.to_datetime(dataframe.date)

    dataset = dataframe[dataframe["date"].dt.year.between(startyear,endyear)]

    dataset.loc[dataset["home_team"] == "Northern Ireland" , "home_team"] = "Ireland"

    dataset.loc[dataset["away_team"] == "Northern Ireland" , "away_team"] = "Ireland"

    dataset.loc[dataset["home_team"] == "Republic of Ireland" , "home_team"] = "Ireland"

    dataset.loc[dataset["away_team"] == "Republic of Ireland" , "away_team"] = "Ireland"

    dataset.loc[dataset["country"] == "Republic of Ireland" , "country"] = "Ireland"

    dataset.loc[dataset["country"] == "Northern Ireland" , "country"] = "Ireland"

    var1 = dataset["home_score"] > dataset["away_score"] 

    var2 = dataset["away_score"] > dataset["home_score"]

    var3 = dataset["home_team"]

    var4 = dataset["away_team"]

    df1 = pd.DataFrame({"hometeam": var3,"awayteam": var4 , "home": var1 , "away" : var2})

    df1.loc[(df1["home"] == True) & (df1["away"] == False) , "home"] = "win"

    df1.loc[(df1["away"] == True) & (df1["home"] == False) , "away"] = "win"

    df1.loc[(df1["home"] == False) & (df1["away"] == False) , ["home" , "away"]] = "draw"

    df1.loc[(df1["home"] == False) & (df1["away"] == "win") , "home"] = "loss"

    df1.loc[(df1["away"] == False) & (df1["home"] == "win") , "away"] = "loss"

    homewinner = df1.loc[df1["home"] == "win" , "hometeam"]

    home_winner = homewinner.value_counts()

    awaywinner = df1.loc[df1["away"] == "win" , "awayteam"]

    homeloser = df1.loc[df1["home"] == "loss" , "hometeam"]

    awayloser = df1.loc[df1["away"] == "loss" , "awayteam"]

    away_winner = awaywinner.value_counts()

    homedraw = df1.loc[df1["home"] == "draw" , "hometeam"]

    awaydraw = df1.loc[df1["away"] == "draw" , "awayteam"]

    home_draw = homedraw.value_counts()

    away_draw = awaydraw.value_counts()

    home_loser = homeloser.value_counts()

    away_loser = awayloser.value_counts()

    df2 = pd.DataFrame({ "homewinner": home_winner , "awaywinner": away_winner , "homedraw": home_draw , "awaydraw" : away_draw , "homelose" : home_loser , "awaylose" : away_loser})

    df2 = df2.fillna(0)

    df2["totalgames"] =  df2["homewinner"] + df2["awaywinner"] + df2["homedraw"] + df2["awaydraw"] + df2["homelose"] + df2["awaylose"]

    df2["totalwins"] = df2["homewinner"] + df2["awaywinner"]

    df2["totalloss"] = df2["homelose"] + df2["awaylose"]

    df2["totaldraw"] = df2["homedraw"] + df2["awaydraw"]

    df2["winnig_percentage"] = (df2["totalwins"] / df2["totalgames"]) * 100

    df2["losing_percentage"] = (df2["totalloss"] / df2["totalgames"]) * 100

    df3 = pd.DataFrame({"Total Games" : df2["totalgames"] , "Total Wins" : df2["totalwins"] , "Total Loss" : df2["totalloss"] ,"Winning Percentage" : df2["winnig_percentage"] , "Total Draw" : df2["totaldraw"] ,  "Losing Percentage" : df2["losing_percentage"]})

    return df3
# Definig a function to get the dataframes upto time ranges

def era(dataframe,startyear,endyear):

    dataframe["date"] = pd.to_datetime(dataframe.date)

    eighteens_to_nineteens = dataframe[dataframe["date"].dt.year.between(startyear,endyear)]

    return eighteens_to_nineteens

era(data,1990,2010)
def extract(dataframe, startyear,endyear):

    dataframe["date"] = pd.to_datetime(dataframe.date)

    dataset = dataframe[dataframe["date"].dt.year.between(startyear,endyear)]

    dataset.loc[dataset["home_team"] == "Northern Ireland" , "home_team"] = "Ireland"

    dataset.loc[dataset["away_team"] == "Northern Ireland" , "away_team"] = "Ireland"

    dataset.loc[dataset["home_team"] == "Republic of Ireland" , "home_team"] = "Ireland"

    dataset.loc[dataset["away_team"] == "Republic of Ireland" , "away_team"] = "Ireland"

    dataset.loc[dataset["country"] == "Republic of Ireland" , "country"] = "Ireland"

    dataset.loc[dataset["country"] == "Northern Ireland" , "country"] = "Ireland"

    var1 = dataset["home_score"] > dataset["away_score"] 

    var2 = dataset["away_score"] > dataset["home_score"]

    var3 = dataset["home_team"]

    var4 = dataset["away_team"]

    df1 = pd.DataFrame({"hometeam": var3,"awayteam": var4 , "home": var1 , "away" : var2})

    df1.loc[(df1["home"] == True) & (df1["away"] == False) , "home"] = "win"

    df1.loc[(df1["away"] == True) & (df1["home"] == False) , "away"] = "win"

    df1.loc[(df1["home"] == False) & (df1["away"] == False) , ["home" , "away"]] = "draw"

    df1.loc[(df1["home"] == False) & (df1["away"] == "win") , "home"] = "loss"

    df1.loc[(df1["away"] == False) & (df1["home"] == "win") , "away"] = "loss"

    homewinner = df1.loc[df1["home"] == "win" , "hometeam"]

    home_winner = homewinner.value_counts()

    awaywinner = df1.loc[df1["away"] == "win" , "awayteam"]

    homeloser = df1.loc[df1["home"] == "loss" , "hometeam"]

    awayloser = df1.loc[df1["away"] == "loss" , "awayteam"]

    away_winner = awaywinner.value_counts()

    homedraw = df1.loc[df1["home"] == "draw" , "hometeam"]

    awaydraw = df1.loc[df1["away"] == "draw" , "awayteam"]

    home_draw = homedraw.value_counts()

    away_draw = awaydraw.value_counts()

    home_loser = homeloser.value_counts()

    away_loser = awayloser.value_counts()

    df2 = pd.DataFrame({ "homewinner": home_winner , "awaywinner": away_winner , "homedraw": home_draw , "awaydraw" : away_draw , "homelose" : home_loser , "awaylose" : away_loser})

    df2 = df2.fillna(0)

    df2["totalgames"] =  df2["homewinner"] + df2["awaywinner"] + df2["homedraw"] + df2["awaydraw"] + df2["homelose"] + df2["awaylose"]

    df2["totalwins"] = df2["homewinner"] + df2["awaywinner"]

    df2["totalloss"] = df2["homelose"] + df2["awaylose"]

    df2["totaldraw"] = df2["homedraw"] + df2["awaydraw"]

    df2["winnig_percentage"] = (df2["totalwins"] / df2["totalgames"]) * 100

    df2["losing_percentage"] = (df2["totalloss"] / df2["totalgames"]) * 100

    df3 = pd.DataFrame({"Total Games" : df2["totalgames"] , "Total Wins" : df2["totalwins"] , "Total Loss" : df2["totalloss"] ,"Winning Percentage" : df2["winnig_percentage"] , "Total Draw" : df2["totaldraw"] ,  "Losing Percentage" : df2["losing_percentage"]})

    return df3
extract(data,2010,2019)