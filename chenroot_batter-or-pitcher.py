# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



import seaborn as sns

import matplotlib.pyplot as plt

sns.set(style="white", context="talk")
df = pd.read_csv("../input/team.csv")

columns_need = ["year", "w", "h", "double", "triple","hr", "bb", "so", "era", "bba", "soa"]

#columns_need = ["year", "w", "hr", "era"]

df_team = df[(df["year"] > 1999) & (df["team_id"] == "BOS")][columns_need]

df_team["1b"] = df_team["h"] - df_team["double"] - df_team["triple"] - df_team["hr"]

df_team = df_team.set_index("year")

'''

print(df_team["1b"].corr(df_team["w"]))

print(df_team["hr"].corr(df_team["w"]))

print(df_team["bb"].corr(df_team["w"]))

print(df_team["so"].corr(df_team["w"]))

print(df_team["bba"].corr(df_team["w"]))

print(df_team["soa"].corr(df_team["w"]))

print(df_team["era"].corr(df_team["w"]))

'''

df_corr = df_team.corr()

#print(df_corr)

df_corr["w"].plot(kind='barh')
df = pd.read_csv("../input/team.csv")

columns_need = ["year", "w", "h", "double", "triple","hr", "bb", "so", "era", "bba", "soa"]

#columns_need = ["year", "w", "hr", "era"]

df_team = df[(df["year"] > 1999) & (df["team_id"] == "NYA")][columns_need]

df_team["1b"] = df_team["h"] - df_team["double"] - df_team["triple"] - df_team["hr"]

df_team = df_team.set_index("year")

'''

print(df_team["1b"].corr(df_team["w"]))

print(df_team["hr"].corr(df_team["w"]))

print(df_team["bb"].corr(df_team["w"]))

print(df_team["so"].corr(df_team["w"]))

print(df_team["bba"].corr(df_team["w"]))

print(df_team["soa"].corr(df_team["w"]))

print(df_team["era"].corr(df_team["w"]))

'''

df_corr = df_team.corr()

#print(df_corr)

df_corr["w"].plot(kind='barh')
df = pd.read_csv("../input/team.csv")

columns_need = ["year", "w", "h", "double", "triple","hr", "bb", "so", "era", "bba", "soa"]

#columns_need = ["year", "w", "hr", "era"]

df_team = df[(df["year"] > 1999) & (df["team_id"] == "OAK")][columns_need]

df_team["1b"] = df_team["h"] - df_team["double"] - df_team["triple"] - df_team["hr"]

df_team = df_team.set_index("year")

'''

print(df_team["1b"].corr(df_team["w"]))

print(df_team["hr"].corr(df_team["w"]))

print(df_team["bb"].corr(df_team["w"]))

print(df_team["so"].corr(df_team["w"]))

print(df_team["bba"].corr(df_team["w"]))

print(df_team["soa"].corr(df_team["w"]))

print(df_team["era"].corr(df_team["w"]))

'''

df_corr = df_team.corr()

#print(df_corr)

df_corr["w"].plot(kind='barh')