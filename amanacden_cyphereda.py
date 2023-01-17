# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



#Import libraries

import numpy as np

import pandas as pd

import sqlite3

from datetime import timedelta

import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

import seaborn as sns

%matplotlib inline

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

from mpl_toolkits.basemap import Basemap

import folium

import folium.plugins

from matplotlib import animation,rc

import io

import base64

import itertools

from subprocess import check_output

import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import pandas as pd

countries_leagues = pd.read_csv("../input/cypherdata/Country_League.csv")

matches = pd.read_csv("../input/cypherdata/Match.csv")

player = pd.read_csv("../input/cypherdata/Player.csv")

player_attributes = pd.read_csv("../input/cypherdata/Player_Attributes.csv")

teams = pd.read_csv("../input/cypherdata/Team.csv")

teams.drop(['Unnamed: 3'],1,inplace=True)

team_attributes = pd.read_csv("../input/cypherdata/Team_Attributes.csv")
matches = matches.reset_index()
matches.rename(columns={'index':'match_api_id'},inplace=True)
countries_leagues.columns = [x.lower() for x in countries_leagues.columns]

team_attributes.columns = [x.lower() for x in team_attributes.columns]

teams.columns = [x.lower() for x in teams.columns]
matches.columns = [x.lower() for x in matches.columns]
matches.columns
matches_new = matches[['country_id', 'season', 'stage', 'date','match_api_id',

                    'home_team_api_id', 'away_team_api_id',

                    'home_team_goal', 'away_team_goal']]
data = matches_new.merge(countries_leagues,left_on="country_id",right_on="country_id",how="outer")
data.rename(columns={'id':'league_id','league_name':'league'},inplace=True)
data.nunique()
plt.figure(figsize=(8,8))

ax = sns.countplot(y = data["league"],

                   order=data["league"].value_counts().index,

                   linewidth = 1,

                   edgecolor = "k"*data["league"].nunique()

                 )

for i,j in enumerate(data["league"].value_counts().values):

    ax.text(.7,i,j,weight = "bold")

plt.title("Matches by league")

plt.show()
data.groupby("league").agg({"home_team_goal":"sum","away_team_goal":"sum"}).plot(kind="barh",

                                                                                 figsize = (10,10),

                                                                                 edgecolor = "k",

                                                                                 linewidth =1

                                                                                )

plt.title("Home and away goals by league")

plt.legend(loc = "best" , prop = {"size" : 14})

plt.xlabel("total goals")

plt.show()
#converting to date format

data["date"] = pd.to_datetime(data["date"],format="%Y-%m-%d")

#extracting year

data["year"] = pd.DatetimeIndex(data["date"]).year
plt.figure(figsize=(10,10))

sns.countplot(y = data["season"],hue=data["league"],

              palette=["r","g","b","c","lime","m","y","k","gold","orange"])

plt.title("MATCHES PLAYED IN EACH LEAGUE BY SEASON")

plt.show()
data = data.merge(teams,left_on="home_team_api_id",right_on="team_api_id",how="left")

data = data.drop(["team_api_id"],axis = 1)

data = data.rename(columns={ 'team_long_name':"home_team_lname",'team_short_name':"home_team_sname"})

data.columns
data = data.merge(teams,left_on="away_team_api_id",right_on="team_api_id",how="left")

data = data.drop(["team_api_id"],axis = 1)

data = data.rename(columns={ 'team_long_name':"away_team_lname",'team_short_name':"away_team_sname"})

data.columns
h_t = data.groupby("home_team_lname")["home_team_goal"].sum().reset_index()

a_t = data.groupby("away_team_lname")["away_team_goal"].sum().reset_index()

h_t = h_t.sort_values(by="home_team_goal",ascending= False)

a_t = a_t.sort_values(by="away_team_goal",ascending= False)

plt.figure(figsize=(13,10))

plt.subplot(121)

ax = sns.barplot(y="home_team_lname",x="home_team_goal",

                 data=h_t[:20],palette="summer",

                 linewidth = 1,edgecolor = "k"*20)

plt.ylabel('')

plt.title("top teams by home goals")

for i,j in enumerate(h_t["home_team_goal"][:20]):

    ax.text(.7,i,j,weight = "bold")

plt.subplot(122)

ax = sns.barplot(y="away_team_lname",x="away_team_goal",

                 data=a_t[:20],palette="winter",

                linewidth = 1,edgecolor = "k"*20)

plt.ylabel("")

plt.subplots_adjust(wspace = .4)

plt.title("top teams by away goals")

for i,j in enumerate(a_t["away_team_goal"][:20]):

    ax.text(.7,i,j,weight = "bold")
x = h_t

x = x.rename(columns={'home_team_lname':"team", 'home_team_goal':"goals"})

y = a_t

y = y.rename(columns={'away_team_lname':"team", 'away_team_goal':"goals"})

goals = pd.concat([x,y])

goals = goals.groupby("team")["goals"].sum().reset_index().sort_values(by = "goals",ascending = False)

plt.figure(figsize=(9,14))

ax = sns.barplot(x="goals",y="team",

                 data=goals[:30],palette="rainbow",

                linewidth = 1,edgecolor = "k"*30)

for i,j in enumerate(goals["goals"][:30]):

    ax.text(.3,i,j,weight="bold",color = "k",fontsize =12)

plt.title("Teams with highest total goals ")

plt.show()
x = data.groupby("home_team_lname")["match_api_id"].count().reset_index()

x = x.rename(columns={"home_team_lname":"team"})

y = data.groupby("away_team_lname")["match_api_id"].count().reset_index()

y = y.rename(columns={"away_team_lname":"team"})

xy = pd.concat([x,y],axis=0)

match_teams =  xy.groupby("team")["match_api_id"].sum().reset_index().sort_values(by="match_api_id",ascending =False)

match_teams = match_teams.rename(columns={"match_api_id":"matches_played"})

match_teams[:20]
#selecting top 50 teams with highest goals

ts = list(goals["team"][:50])

v =data[["home_team_lname","away_team_lname"]]

v = v[(v["home_team_lname"].isin(ts)) & (v["away_team_lname"].isin(ts))]

import networkx as nx

g = nx.from_pandas_edgelist(v,"home_team_lname","away_team_lname")

fig = plt.figure(figsize=(10,10))

nx.draw_kamada_kawai(g,with_labels =True,node_size =2500,node_color ="Orangered",alpha=.8)

plt.title("NETWORK LAYOUT FOR MATCHES PLAYED BETWEEN TOP SCORERS")

fig.set_facecolor("white")
ts1 = ['Club Brugge KV','Olympique Lyonnais','Arsenal','Celtic','Borussia Dortmund','PSV','FC Bayern Munich','Legia Warszawa','SL Benfica','Manchester City','Paris Saint-Germain','Roma','Juventus','Real Madrid CF','FC Barcelona','FC Basel']
plt.figure(figsize=(12,6))

sns.kdeplot(data["home_team_goal"],shade=True,

            color="b",label="home goals")

sns.kdeplot(data["away_team_goal"],shade=True,

            color="r",label="away goals")

plt.axvline(data["home_team_goal"].mean(),linestyle = "dashed",

            color="b",label="home goals mean")

plt.axvline(data["away_team_goal"].mean(),linestyle = "dashed",

            color="r",label="away goals mean")

plt.legend(loc="best",prop = {"size" : 12})

plt.title("DISTRIBUTION OF HOME AND AWAY GOALS")

plt.xlabel("goals")

plt.show()
data.groupby(["home_team_lname","league"]).agg({"match_api_id":"count","home_team_goal":"sum"}).reset_index()
v =data[["home_team_lname","away_team_lname"]]

v = v[(v["home_team_lname"].isin(ts1)) & (v["away_team_lname"].isin(ts1))]
x = data.groupby(["home_team_lname","league"]).agg({"match_api_id":"count","home_team_goal":"sum"}).reset_index()

y = data.groupby(["away_team_lname","league"]).agg({"match_api_id":"count","away_team_goal":"sum"}).reset_index()

x = x.rename(columns={'home_team_lname':"team", 'match_api_id':"matches", 'home_team_goal':"goals"})

y = y.rename(columns={'away_team_lname':"team", 'match_api_id':"matches", 'away_team_goal':"goals"})

xy = pd.concat([x,y])

xy = xy.groupby(["team","league"])[["matches","goals"]].sum().reset_index()

xy = xy.sort_values(by="goals",ascending=False)

plt.figure(figsize=(13,6))

c   = ["r","g","b","m","y","yellow","c","orange","grey","lime","white"]

lg = xy["league"].unique()

for i,j,k in itertools.zip_longest(lg,range(len(lg)),c):

    plt.scatter("matches","goals",data=xy[xy["league"] == i],label=[i],s=100,alpha=1,linewidths=1,edgecolors="k",color=k)

    plt.legend(loc="best")

    plt.xlabel("MATCHES")

    plt.ylabel("GOALS SCORED")



plt.title("MATCHES VS GOALS BY TEAMS")

plt.show()
x = data[data["home_team_lname"].isin(ts1)].groupby(["home_team_lname","league"]).agg({"match_api_id":"count","home_team_goal":"sum"}).reset_index()

y = data[data["away_team_lname"].isin(ts1)].groupby(["away_team_lname","league"]).agg({"match_api_id":"count","away_team_goal":"sum"}).reset_index()

x = x.rename(columns={'home_team_lname':"team", 'match_api_id':"matches", 'home_team_goal':"goals"})

y = y.rename(columns={'away_team_lname':"team", 'match_api_id':"matches", 'away_team_goal':"goals"})

xy = pd.concat([x,y])

xy = xy.groupby(["team","league"])[["matches","goals"]].sum().reset_index()

xy = xy.sort_values(by="goals",ascending=False)

plt.figure(figsize=(13,6))

c   = ["r","g","b","m","y","yellow","c","orange","grey","lime","white"]

lg = xy["league"].unique()

for i,j,k in itertools.zip_longest(lg,range(len(lg)),c):

    plt.scatter("matches","goals",data=xy[xy["league"] == i],label=[i],s=100,alpha=1,linewidths=1,edgecolors="k",color=k)

    plt.legend(loc="best")

    plt.xlabel("MATCHES")

    plt.ylabel("GOALS SCORED")



plt.title("MATCHES VS GOALS BY TEAMS")

plt.show()
plt.figure(figsize=(8,12))

plt.scatter(y = xy["team"][:6],x = xy["matches"][:6],

            s=xy["goals"],alpha=.7,c=sns.color_palette("Blues"),

            linewidths=1,edgecolors="b")

plt.xticks(rotation = 90)

plt.xlabel("matchess played")

plt.title("MATCHES VS GOALS BY TOP 50 TEAMS")

plt.show()
plt.figure(figsize=(13,10))

plt.subplot(211)

sns.boxplot(x = data["season"],y = data["away_team_goal"],palette="rainbow")

plt.title("HOME GOALS BY SEASON")

plt.subplot(212)

sns.boxplot(x = data["season"],y = data["home_team_goal"],palette="rainbow")

plt.title("AWAY GOALS BY SEASON")

plt.show()

data["total_goal"] = data["home_team_goal"]+data["away_team_goal"]

a = data.groupby("season").agg({"total_goal":"sum"})

m = data.groupby("season").agg({"total_goal":"mean"})

s = data.groupby("season").agg({"total_goal":"std"})

x = data.groupby("season").agg({"total_goal":"max"})

xx = a.merge(m,left_index=True,right_index=True,how="left")

yy = s.merge(x,left_index=True,right_index=True,how="left")

x_y = xx.merge(yy,left_index=True,right_index=True,how="left").reset_index()

x_y = x_y.rename(columns={'total_goal_x_x':"goals", 'total_goal_y_x':"mean",

                          'total_goal_x_y':"std",'total_goal_y_y':"max"})

import itertools

cols = [ 'goals', 'mean', 'std', 'max' ]

length = len(cols)

cs   = ["r","g","b","c"] 

plt.figure(figsize=(12,16))



for i,j,k in itertools.zip_longest(cols,range(length),cs):

    plt.subplot(length,length/length,j+1)

    sns.pointplot(x_y["season"],x_y[i],color=k)

    plt.title(i)

    plt.subplots_adjust(hspace =.3)
g = nx.from_pandas_edgelist(data[(data["home_team_lname"].isin(ts)) & (data["away_team_lname"].isin(ts))],"home_team_sname","away_team_sname")

fig = plt.figure(figsize=(11,11))

nx.draw_kamada_kawai(g,with_labels = True)

plt.title("INTERACTION BETWEEN TEAMS")

fig.set_facecolor("white")
#create new feature for winning team

def label(data):

    if data["home_team_goal"] > data["away_team_goal"]:

        return data["home_team_lname"]

    elif data["away_team_goal"] > data["home_team_goal"]:

        return data["away_team_lname"]

    elif data["home_team_goal"] == data["away_team_goal"]:

        return "DRAW"
data["win"] = data.apply(lambda data:label(data),axis=1)

#create new feature for outcome of match

def lab(data):

    if data["home_team_goal"] > data["away_team_goal"]:

        return "HOME TEAM WIN"

    elif data["away_team_goal"] > data["home_team_goal"]:

        return "AWAY TEAM WIN"

    elif data["home_team_goal"] == data["away_team_goal"]:

        return "DRAW"
data["outcome_side"] = data.apply(lambda data:lab(data),axis = 1)
#create new feature for losing team

def labe(data):

    if data["home_team_goal"] < data["away_team_goal"]:

        return data["home_team_lname"]

    elif data["away_team_goal"] < data["home_team_goal"]:

        return data["away_team_lname"]

    elif data["home_team_goal"] == data["away_team_goal"]:

        return "DRAW"

    
data["lost"] = data.apply(lambda data:labe(data),axis=1)
plt.figure(figsize=(8,8))

data["outcome_side"].value_counts().plot.pie(autopct = "%1.0f%%",

                                             colors =sns.color_palette("rainbow",3),

                                             wedgeprops = {"linewidth":2,"edgecolor":"white"})

my_circ = plt.Circle((0,0),.7,color = "white")

plt.gca().add_artist(my_circ)

plt.title("PROPORTION OF GAME OUTCOMES")

plt.show()
win = data[(data["home_team_lname"].isin(ts1)) & (data["away_team_lname"].isin(ts1))]["win"].value_counts()[1:].reset_index()

lost = data[(data["home_team_lname"].isin(ts1)) & (data["away_team_lname"].isin(ts1))]["lost"].value_counts()[1:].reset_index()

plt.figure(figsize=(13,14))

plt.subplot(121)

ax = sns.barplot(win["win"],win["index"],

                 palette="Set2",

                linewidth = 1,edgecolor = "k"*30)

plt.title(" TOP WINNING TEAMS")

plt.ylabel("")

for i,j in enumerate(win["win"]):

    ax.text(.7,i,j,color = "black",weight = "bold")

    

plt.subplot(122)

ax = sns.barplot(lost["lost"][:30],lost["index"][:30],

                 palette="Set2",

                linewidth = 1,edgecolor = "k"*30)

plt.title(" TOP TEAMS that Lost")

plt.subplots_adjust(wspace = .3)

plt.ylabel("")

for i,j in enumerate(lost["lost"][:30]):

    ax.text(.7,i,j,color = "black",weight = "bold")
ts1
#merge win,draw and lost data of team to matches played

f = xy.merge(win,left_on="team",right_on="index",how="left")

f = f.drop("index",axis =1)

f = f.rename(columns={"outcome":"wins"})

f = f.merge(lost,left_on="team",right_on="index",how="left")

f = f.drop("index",axis =1)

dr = data[data["outcome_side"] == "DRAW"][["home_team_lname","away_team_lname"]]

l  = dr["home_team_lname"].value_counts().reset_index()

v  = dr["away_team_lname"].value_counts().reset_index()

l  = l.rename(columns={'index':"team", 'home_team_lname':"draw"})

v  = v.rename(columns={'index':"team", 'away_team_lname':"draw"})

lv = pd.concat([l,v])

lv = lv.groupby("team")["draw"].sum().reset_index()

f = f.merge(lv,left_on="team",right_on="team",how ="left")
f = f.sort_values(by="goals",ascending=False)

f_new = f.copy()

f_new.index = f_new.team

f_new[["win","lost","draw"]][:20].plot(kind = "bar",figsize=(13,5),

                                   stacked =True,linewidth = 1,

                                   edgecolor = "k"*20

                                  )

plt.legend(loc="best",prop = {"size" : 13})

plt.title("PERFORMANCE BY TOP TEAMS")

plt.ylabel("matches played")

plt.show()
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10,10))

ax  = fig.add_subplot(111,projection ="3d")

ax.scatter(f["win"],f["lost"],f["draw"],s=f["matches"]*3,

           alpha=.4,linewidth =1,edgecolor= "k",c = "lime")

ax.set_xlabel("wins")

ax.set_ylabel("lost")

ax.set_zlabel("draw")

plt.title("WIN VS LOST VS DRAW")

plt.show()

plt.figure(figsize=(13,7))

cols = ["matches","win","lost","draw"]

c    = ["b","orange","lime","m"]

length = len(cols)

for i,j,k in itertools.zip_longest(cols,range(length),c):

    plt.stackplot(f.index,f[i],alpha=.6,color = k,labels=[i])

    plt.axhline(f[i].mean(),color=k,

                linestyle="dashed",label=i+ " mean")

    plt.legend(loc="best")

    plt.title("AREA PLOT FOR MATCH ATTRIBUTES")

    plt.xlabel("team index")
pd.DataFrame(data.groupby(["league"])["win"].count())
pd.DataFrame(data.groupby(["home_team_sname"])["win"].count())
data
x = pd.DataFrame(data.groupby(["league","win"])["win"].count())

x = x.rename(columns={"win":"team"}).reset_index()

x = x.rename(columns={"win":"team","team":"win"})

x = x.sort_values(by="win",ascending=False)

x = x[x["team"] != "DRAW"]

x = x.drop_duplicates(subset=["league"],keep="first")

plt.figure(figsize=(8,7))

ax =sns.barplot(x["win"],x["league"],palette="cool",

               linewidth = 1 ,edgecolor = "k"*10)

for i,j in enumerate(x["team"]):

    ax.text(.7,i,j,weight = "bold",fontsize = 12)

plt.title("TOP TEAMS BY LEAGUES")

plt.show()
data.groupby(["league"]).agg({"match_api_id":"count","total_goal":"sum"}).plot(kind="barh",

                                                                               stacked =True,

                                                                               figsize=(8,8),

                                                                               linewidth = 1,

                                                                               edgecolor = "k"*data["league"].nunique()

                                                                              )

plt.title("# MATCHES PLAYED IN EACH LEAGUE VS TOTAL GOALS SCORED")

plt.show()
plt.figure(figsize=(7,15))

plt.subplot(211)

agg = data.groupby(["league"]).agg({"match_api_id":"count","total_goal":"sum"})

agg["match_api_id"].plot.pie(colors=sns.color_palette("seismic",10),

                             autopct="%1.0f%%",

                             wedgeprops={"linewidth":2,"edgecolor":"white"})

plt.ylabel("")

my_circ = plt.Circle((0,0),.7,color ="white")

plt.gca().add_artist(my_circ)

plt.title("PROPORTION OF MATCHES PLAYED IN LEAGUES")

plt.subplot(212)

agg["total_goal"].plot.pie(colors=sns.color_palette("seismic",10),

                           autopct="%1.0f%%",

                           wedgeprops={"linewidth":2,"edgecolor":"white"})

plt.ylabel("")

my_circ = plt.Circle((0,0),.7,color ="white")

plt.gca().add_artist(my_circ)

plt.title("PROPORTION OF GOALS SCORED IN LEAGUES")

plt.show()
from wordcloud import WordCloud

import nltk

wrd = data[data["win"] != "DRAW"]["win"].to_frame()

wrd = wrd["win"].value_counts()[wrd["win"].value_counts() > 100].keys().str.replace(" ","")

wrd = pd.DataFrame(wrd)

wc = WordCloud(background_color="black",scale =2,colormap="flag").generate(str(wrd[0]))

plt.figure(figsize=(13,8))

plt.imshow(wc,interpolation="bilinear")

plt.axis("off")

plt.title("TOP TEAMS")

plt.show()
pvt = pd.pivot_table(index="season",columns="league",values="total_goal",data=data,aggfunc="sum")

pvt.plot(kind = "barh",stacked = True,figsize =(10,8),

         colors =sns.color_palette("rainbow",11),

         linewidth = .5,edgecolor = ["grey"]*10)

plt.title("GOALS SCORED IN EACH SEASON OF LEAUGES")

plt.show()
i = data["win"].value_counts()[1:25].index

t= pd.pivot_table(index="home_team_lname",columns="season",values="home_team_goal",

                  data=data,aggfunc="sum")

t=t[t.index.isin(i)]

t.plot(kind="barh",stacked=True,figsize=(10,10),colors=sns.color_palette("prism",11))

plt.title("HOME GOALS SCORED BY TOP TEAMS BY SEASON")

plt.show()
i = data["win"].value_counts()[1:25].index

t= pd.pivot_table(index="away_team_lname",columns="season",

                  values="away_team_goal",data=data,aggfunc="sum")

t=t[t.index.isin(i)]

t.plot(kind="barh",stacked=True,figsize=(10,10),colors=sns.color_palette("prism",11))

plt.title("AWAY GOALS SCORED BY TOP TEAMS BY SEASON")

plt.show()
nw = data[["season","league","win"]]

nw["team"] = nw["win"]

nw = nw.groupby(["season","league","team"])["win"].count().reset_index().sort_values(by=["season","league","win"],ascending =False)

nw = nw[nw["team"] != "DRAW"]

nw = nw.drop_duplicates(subset=["season","league"],keep="first").sort_values(by=["league","season"],ascending =True)



plt.figure(figsize=(13,28))

plt.subplot(621)

lg = nw[nw["league"] == "Belgium Jupiler League"]

ax = sns.barplot(lg["win"],lg["season"],palette="cool",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title("Belgium Jupiler League")

plt.xlabel("")

plt.ylabel("")



plt.subplot(622)

lg = nw[nw["league"] == "England Premier League"]

ax = sns.barplot(lg["win"],lg["season"],palette="magma",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold",color="white")

plt.title("England Premier League")

plt.xlabel("")

plt.ylabel("")



plt.subplot(623)

lg = nw[nw["league"] == 'Spain LIGA BBVA']

ax = sns.barplot(lg["win"],lg["season"],palette="rainbow",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Spain LIGA BBVA')

plt.xlabel("")

plt.ylabel("")



plt.subplot(624)

lg = nw[nw["league"] == 'France Ligue 1']

ax = sns.barplot(lg["win"],lg["season"],palette="summer",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold",color = "white")

plt.title('France Ligue 1')

plt.xlabel("")

plt.ylabel("")



plt.subplot(625)

lg = nw[nw["league"] == 'Germany 1. Bundesliga']

ax = sns.barplot(lg["win"],lg["season"],palette="winter",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Germany 1. Bundesliga')

plt.xlabel("")

plt.ylabel("")



plt.subplot(626)

lg = nw[nw["league"] == 'Italy Serie A']

ax = sns.barplot(lg["win"],lg["season"],palette="husl",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Italy Serie A')

plt.xlabel("")

plt.ylabel("")

plt.show()
plt.figure(figsize=(13,28))

plt.subplot(621)

lg = nw[nw["league"] == 'Netherlands Eredivisie']

ax = sns.barplot(lg["win"],lg["season"],palette="Blues",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Netherlands Eredivisie')

plt.xlabel("")

plt.ylabel("")



plt.subplot(622)

lg = nw[nw["league"] == 'Poland Ekstraklasa']

ax = sns.barplot(lg["win"],lg["season"],palette="winter",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Poland Ekstraklasa')

plt.xlabel("")

plt.ylabel("")



plt.subplot(623)

lg = nw[nw["league"] == 'Portugal Liga ZON Sagres']

ax = sns.barplot(lg["win"],lg["season"],palette="rainbow",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Portugal Liga ZON Sagres')

plt.xlabel("")

plt.ylabel("")



plt.subplot(624)

lg = nw[nw["league"] == 'Scotland Premier League']

ax = sns.barplot(lg["win"],lg["season"],palette="Greens",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Scotland Premier League')

plt.xlabel("")

plt.ylabel("")



plt.subplot(625)

lg = nw[nw["league"] == 'Switzerland Super League']

ax = sns.barplot(lg["win"],lg["season"],palette="cool",

                linewidth = 1 ,edgecolor = "k"*lg["season"].nunique())

for i,j in enumerate(lg["team"]):

    ax.text(.7,i,j,weight = "bold")

plt.title('Switzerland Super League')

plt.xlabel("")

plt.ylabel("")

plt.show()
#Extract year

player["year"]  = pd.DatetimeIndex(player["birthday"]).year

#extract age

player["age"]   = 2019 - player["year"]
#merge player data with player attributes

player_info = player_attributes.merge(player,left_on="player_api_id",right_on="player_api_id",how="left")
i =["id_x","id_y",'player_fifa_api_id_y','height', 'weight', 'weight_kg', 'height_m', 'bmi', 'year','age','birthday']

player_info = player_info[[x for x in player_info.columns if x not in i]]

player_info.columns
player_info["date"] = pd.to_datetime(player_info["date"],format="%Y-%m-%d")

ax = player_info["player_name"].value_counts().sort_values()[-20:].plot(kind="barh",figsize=(10,10),

                                                                        color="b",width=.9,

                                                                        linewidth = 1,edgecolor = "k"*20

                                                                       )

for i,j in enumerate(player_info["player_name"].value_counts().sort_values()[-20:].values):

    ax.text(.7,i,j,weight = "bold",color="white")

ax.set_title("PLAYERS WHO PLAYED HIGHEST MATCHES")

plt.show()
teams['team_long_name']
data
play = player_info[player_info["overall_rating"]  > 86 ]["player_name"].value_counts().index

import nltk

from PIL import Image

img = np.array(Image.open("/kaggle/input/picturewrd/z.jpg"))

wc = WordCloud(background_color="black",scale=2,mask=img,colormap="cool",max_words=100000).generate(" ".join(play))

fig = plt.figure(figsize=(15,15))

plt.imshow(wc,interpolation="bilinear")

plt.axis("off")

plt.title("WORD CLOUD FOR PLAYER NAMES")

plt.show()
player_info[player_info["overall_rating"]  > 88 ]["player_name"].value_counts().index
top_rated = player_info[player_info["overall_rating"]  > 88 ]["player_name"].value_counts().index

import nltk

wc = WordCloud(background_color="white",scale=2).generate(" ".join(top_rated))

fig = plt.figure(figsize=(15,8))

plt.imshow(wc,interpolation="bilinear")

plt.axis("off")

plt.title("TOP RATED PLAYERS")

plt.show()
team_info =  team_attributes.merge(teams,left_on="team_api_id",right_on="team_api_id",how="left")
team_info["date"] = pd.to_datetime(team_info["date"],format="%Y-%m-%d")
columns= team_info.columns

cat_col= columns[columns.str.contains("class")].tolist()
num_col= [x for x in team_info.columns if x not in columns[columns.str.contains("class")].tolist()+["team_api_id"]+["date"]+['team_long_name']+['team_short_name']]
categorical_team_info = team_info[cat_col+["team_api_id"]+["date"]+['team_long_name']+[ 'team_short_name']]
numerical_team_info   = team_info[num_col+["team_api_id"]+["date"]+['team_long_name']+[ 'team_short_name']]
numerical_team_info

n = numerical_team_info.groupby("team_long_name")[num_col].mean().reset_index()

cols = [x for x in n.columns if x not in ["team_long_name"]]

length = len(cols)

plt.figure(figsize=(13,13))

for i,j in itertools.zip_longest(cols,range(length)):

    plt.subplot(length/3,length/3,j+1)

    ax = sns.barplot(i,"team_long_name",data=n.sort_values(by=i,ascending=False)[:7],palette="winter")

    plt.title(i)

    plt.subplots_adjust(wspace = .6,hspace =.3)

    plt.ylabel("")

    for i,j in enumerate(round(n.sort_values(by = i,ascending=False)[i][:7],2)):

        ax.text(.7,i,j,weight = "bold",color="white") 
categorical_team_info.columns
cat_col.remove('buildupplaypositioningclass')

cat_col.remove('chancecreationpositioningclass')

cat_col.remove('defencedefenderlineclass')
cat_col
from scipy.stats import mode



c = categorical_team_info.groupby("team_long_name").agg({"buildupplayspeedclass":lambda x:mode(x)[0],

                                                    "buildupplaydribblingclass":lambda x:mode(x)[0],

                                                    'buildupplaypassingclass':lambda x:mode(x)[0],

                                                    'chancecreationpassingclass':lambda x:mode(x)[0],

                                                    'chancecreationcrossingclass':lambda x:mode(x)[0],

                                                     'chancecreationshootingclass':lambda x:mode(x)[0],

                                                     'defencepressureclass':lambda x:mode(x)[0],

                                                     'defenceaggressionclass':lambda x:mode(x)[0],

                                                     'defenceteamwidthclass':lambda x:mode(x)[0]

                                                        }).reset_index()

cat_col

flatui = ["#2ecc71","#224c3c", "#11495e"]

plt.figure(figsize=(10,10))

for i,j in itertools.zip_longest(cat_col,range(len(cat_col))):

    plt.subplot(3,3,j+1)

    plt.pie(c[i].value_counts().values,labels=c[i].value_counts().keys(),

            wedgeprops={"linewidth":3,"edgecolor":"w"},

           colors=sns.color_palette(flatui),autopct = "%1.0f%%")

    my_circ = plt.Circle((0,0),.7,color="white")

    plt.gca().add_artist(my_circ)

    plt.title(i)

    plt.xlabel("")
from math import pi

def team_comparator(team1,team2):

    team_list = [team1,team2]

    length    = len(team_list)

    cr        = ["b","r"]

    fig = plt.figure(figsize=(15,8))

    plt.subplot(111,projection= "polar")

    

    for i,j,k in itertools.zip_longest(team_list,range(length),cr):

        cats = num_col

        N    = len(cats)

        

        values = n[n["team_long_name"] ==  i][cats].values.flatten().tolist()

        values += values[:1]

        

        angles = [n/float(N)*2*pi for n in range(N)]

        angles += angles[:1]

        

        plt.xticks(angles[:-1],cats,color="k",fontsize=15)

        plt.plot(angles,values,linewidth=3,color=k)

        plt.fill(angles,values,color = k,alpha=.4,label = i)

        plt.legend(loc="upper right",frameon =True,prop={"size":15}).get_frame().set_facecolor("lightgrey")

        fig.set_facecolor("w")

        fig.set_edgecolor("k")

        plt.title("",fontsize=30,color="tomato")

team_comparator("SL Benfica","Celtic")
x = data.groupby(["home_team_lname","league"]).agg({"match_api_id":"count","home_team_goal":"sum"}).reset_index()

y = data.groupby(["away_team_lname","league"]).agg({"match_api_id":"count","away_team_goal":"sum"}).reset_index()

x = x.rename(columns={'home_team_lname':"team", 'match_api_id':"matches", 'home_team_goal':"goals"})

y = y.rename(columns={'away_team_lname':"team", 'match_api_id':"matches", 'away_team_goal':"goals"})

xy = pd.concat([x,y])

xy = xy.groupby(["team","league"])[["matches","goals"]].sum().reset_index()

xy = xy.sort_values(by="goals",ascending=False)

plt.figure(figsize=(13,6))

c   = ["r","g","b","m","y","yellow","c","orange","grey","lime","white"]

lg = xy["league"].unique()

for i,j,k in itertools.zip_longest(lg,range(len(lg)),c):

    plt.scatter("matches","goals",data=xy[xy["league"] == i],label=[i],s=100,alpha=1,linewidths=1,edgecolors="k",color=k)

    plt.legend(loc="best")

    plt.xlabel("MATCHES")

    plt.ylabel("GOALS SCORED")



plt.title("MATCHES VS GOALS BY TEAMS")

plt.show()