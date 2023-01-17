import numpy as np 

import pandas as pd

import os

print(os.listdir("../input"))
df = pd.read_csv("../input/nfl-draft-outcomes/nfl_draft.csv")

print(df.columns)

pd.unique(df['Pos'])
df=df[["Year",'Rnd','Pick','Tm','Player','Pos']]

df=df[df['Rnd'] == 1]

df['Year'] = df['Year'] - 1

df.head()
df["Nickel Pos"] = df['Pos']

df["Nickel Pos"][df["Pos"] == "OLB"] = "EDGE"

df["Nickel Pos"][df["Pos"] == "T"] = "OT"

df["Nickel Pos"][df["Pos"] == "DE"] = "EDGE"

df["Nickel Pos"][df["Pos"] == "NT"] = "IDL"

df["Nickel Pos"][df["Pos"] == "DT"] = "IDL"

df["Nickel Pos"][df["Pos"] == "C"] = "IOL"

df["Nickel Pos"][df["Pos"] == "G"] = "IOL"

df["Nickel Pos"][df["Pos"] == "ILB"] = "LB"

df["Nickel Pos"][df["Pos"] == "FS"] = "DB"

df["Nickel Pos"][df["Pos"] == "SS"] = "DB"

df["Nickel Pos"][df["Pos"] == "CB"] = "DB"

df.head()
df['Tm'].replace("TEN","OTI",inplace=True)

df['Tm'].replace("PHO","CRD",inplace=True)

df['Tm'].replace("ARI","CRD",inplace=True)

df['Tm'].replace("BAL","RAV",inplace=True)

#df[df['Year']>2000]['Tm'].replace('HOU',"HTX",inplace=True)

df.loc[((df['Year'].apply(int) > 2000) & (df['Tm'] == "HOU")), 'Tm'] = "HTX"

#df[df['Year']<2000]['Tm'].replace('HOU',"OTI",inplace=True)

df.loc[((df['Year'].apply(int) < 2000) & (df['Tm'] == "HOU")), 'Tm'] = "OTI"

#df[df['Year']<1990]['Tm'].replace('STL',"CRD",inplace=True)

df.loc[((df['Year'].apply(int) < 1990) & (df['Tm'] == "STL")), 'Tm'] = "CRD"

#df[df['Year']>1990]['Tm'].replace('STL',"RAM",inplace=True)

df.loc[((df['Year'].apply(int) > 1990) & (df['Tm'] == "STL")), 'Tm'] = "RAM"

df['Tm'].replace("IND","CLT",inplace=True)

df['Tm'].replace("OAK","RAI",inplace=True)



df['Year'] = df['Year'].apply(str)

df['index'] = df['Tm'] + df['Year']

df = df.set_index('index')

df.head()
df1 = pd.read_excel("../input/draft-1strnd-16-18/draft_1strnd_16_18.xlsx")

df1['Year'] = df1['Year'] - 1



df1["Nickel Pos"] = df1['Pos']

df1["Nickel Pos"][df1["Pos"] == "OLB"] = "EDGE"

df1["Nickel Pos"][df1["Pos"] == "T"] = "OT"

df1["Nickel Pos"][df1["Pos"] == "DE"] = "EDGE"

df1["Nickel Pos"][df1["Pos"] == "NT"] = "IDL"

df1["Nickel Pos"][df1["Pos"] == "DT"] = "IDL"

df1["Nickel Pos"][df1["Pos"] == "C"] = "IOL"

df1["Nickel Pos"][df1["Pos"] == "G"] = "IOL"

df1["Nickel Pos"][df1["Pos"] == "ILB"] = "LB"

df1["Nickel Pos"][df1["Pos"] == "FS"] = "DB"

df1["Nickel Pos"][df1["Pos"] == "SS"] = "DB"

df1["Nickel Pos"][df1["Pos"] == "CB"] = "DB"



df1["Year"] = df1["Year"].apply(str)

df1['index'] = df1['Tm'] + df1['Year']

df1 = df1.set_index('index')

df1.rename(columns={"Round":"Rnd"},inplace=True)

draft = df.append(df1)

draft.head()
### this was scraped in the following way:

# from sportsreference.nfl.teams import Teams

# teams = Teams()

# df_blank = teams("BUF").dataframe

# df_blank.drop("BUF",inplace=True)

# df_blank['Year'] = ''

# df = df_blank

# for i in range(1984,2019):

#   teams = Teams(year=i)

#   for j in teams:

#     row = j.dataframe

#     row["Year"] = str(i)

#     df = df.append(row)

# Credits go to pro-football-reference.com twitter: @pfref





seasons = pd.read_excel("../input/nfl-seasons-pfr/nflv2.xlsx",index_col=0)

seasons.head()
print(seasons['abbreviation'].unique())

print(draft['Tm'].unique())
draft = draft[['Tm',"Nickel Pos",'Player']]

draft.head()

season_draft = draft.join(seasons,how='outer')
train = season_draft.dropna(subset=['Nickel Pos'])

train.dropna(how='any',inplace=True)
train_X = train[['defensive_simple_rating_system','fumbles','interceptions',

       'margin_of_victory','offensive_simple_rating_system','pass_completions',

       'pass_net_yards_per_attempt','points_against',

       'rank','rush_first_downs', 'rush_touchdowns',

       'rush_yards_per_attempt','turnovers']]

train_y = train[['Nickel Pos']]
from sklearn.tree import DecisionTreeClassifier



clf = DecisionTreeClassifier(max_depth=3)

clf.fit(train_X,train_y)
pred_X = season_draft[season_draft['Year'] == 2018][['defensive_simple_rating_system','fumbles','interceptions',

       'margin_of_victory','offensive_simple_rating_system','pass_completions',

       'pass_net_yards_per_attempt','points_against',

       'rank','rush_first_downs', 'rush_touchdowns',

       'rush_yards_per_attempt','turnovers']]

pred_X.drop(['CHI2018','CLE2018','DAL2018','NOR2018'],inplace=True)

pred_X['First Draft Pick 2019'] = [14,9,16,11,26,1,10,8,12,23,6,29,13,18,32,7,3,19,25,20,4,31,22,28,21,2,5,15]

pred_X.sort_values(by='First Draft Pick 2019',inplace=True)

pred_X.drop('First Draft Pick 2019',axis=1,inplace=True)

pred_X.head()
clf.predict(pred_X)
### learned this from https://medium.com/@rnbrown/creating-and-visualizing-decision-trees-with-python-f8e8fa394176

from sklearn.externals.six import StringIO  

from IPython.display import Image  

from sklearn.tree import export_graphviz

import pydotplus

dot_data = StringIO()

export_graphviz(clf, out_file=dot_data,  

                filled=True, rounded=True,

                special_characters=True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())
from sklearn.ensemble import RandomForestClassifier



clf2 = RandomForestClassifier(n_estimators=100)

clf2.fit(train_X,train_y)

clf2.predict(pred_X)