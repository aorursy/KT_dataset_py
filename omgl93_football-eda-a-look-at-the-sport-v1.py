import pandas as pd

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import datetime

%matplotlib inline
df= pd.read_csv("../input/international-football-results-from-1872-to-2017/results.csv")
df.isnull().sum()
df.info() #We will use .info to check the data type of the columns.
df.describe() #Describe will help us to a broad outlier check on the data.
df.head()
above_20 = df[df["home_score"] >= 20]

above_20_a = df[df["away_score"] >= 20]
above_20.head(15)
above_20_a.head(15)
def winner(row): #Function for wins

    if row["home_score"] > row["away_score"]:

        return row["home_team"]

    if row["home_score"] == row["away_score"]:

        return "Draw"

    if row["home_score"] < row["away_score"]:

        return row["away_team"]

def loser (row): #Funcion for lose

    if row["home_score"] > row["away_score"]:

        return row["away_team"]

    if row["home_score"] == row["away_score"]:

        return "Draw"

    if row["home_score"] < row["away_score"]:

        return row["home_team"]
df["Winner"] = df.apply (lambda row: winner(row), axis=1)

df["Loser"] = df.apply (lambda row: loser(row), axis=1)
df["date"] = pd.to_datetime(df["date"]) # Converting the colum to datetime
df["Year"] = df["date"].dt.year #Extracting the year from the date column

df = df[df["Year"] < 2019] #Removing 2019 from the mix
scores = df.groupby("date")["home_score","away_score"].agg(["sum","count"]) #Using groupby to get the data we need
scores.info()
scores.columns = ["Home_Number_goals", "Home_Games","Away_Number_goals", "Away_Games"] #Renaming the columns
scores_year = pd.DataFrame(scores["Home_Games"].resample("Y").sum())
scores_goals_h = pd.DataFrame(scores["Home_Number_goals"].resample("Y").sum())

scores_goals_a = pd.DataFrame(scores["Away_Number_goals"].resample("Y").sum())
scores_goals_a.head()
fig, ax = plt.subplots(figsize=(17, 10))

plt.style.use('seaborn-darkgrid')

ax.plot(scores_year["Home_Games"], label="Games per year", color="black")

ax.tick_params(labelsize=12)

plt.legend(loc=0, fontsize="large")

fig.suptitle("Games per Year", fontsize=20)





ax.annotate("Start of WW2", xy=('1939', 105),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-30, -60), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("End of WW2", xy=('1945', 35),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(35, -30), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("Start of WW1", xy=('1914', 35),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-50, 20), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("End of WW1", xy=('1918', 30),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-35, -30), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("Founding of FIFA", xy=('1930', 85),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(35, 60), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("Founding of AFC", xy=('1956', 150),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-35, 60), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("Founding of AFCON", xy=('1957', 200),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(50, -50), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("Founding of CONMEBOL", xy=('1916', 30),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(10, 50), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("FIFA-Expansion to 24 teams", xy=('1982', 500),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-50, 80), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("FIFA-Expansion to 32 teams", xy=('1998', 850),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(-90, 80), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))

ax.annotate("FIFA-Expansion to 48 teams", xy=('2013', 1050),  xycoords='data',size=15,

            bbox=dict(boxstyle="round", fc="none", ec="gray"),

            xytext=(0, -180), textcoords='offset points', ha='center',

            arrowprops=dict(arrowstyle="->"))
#5 day SMA

fig, ax = plt.subplots(figsize=(14, 8))

ax.plot(scores_goals_h, label="Mean home goals")

ax.plot(scores_goals_a, label="Mean away goals")

ax.plot(scores_goals_h.rolling(window=5).mean(), color = "red", label="Rolling mean of home goals")

ax.plot(scores_goals_a.rolling(window=5).mean(), color = "red", label="Rolling mean of away goals")

plt.legend(loc=0, fontsize="large")

fig.suptitle("Mean goals per year", fontsize=20)

plt.legend()
h_goals_year = pd.DataFrame(scores["Home_Number_goals"].resample("10A").sum())

a_goals_year = pd.DataFrame(scores["Away_Number_goals"].resample("10A").sum())
h_goals_year["Decade"] = h_goals_year.index

h_goals_year["Decade"] = h_goals_year["Decade"].dt.year

a_goals_year["Decade"] = a_goals_year.index

a_goals_year["Decade"] = a_goals_year["Decade"].dt.year
fig,ax = plt.subplots(figsize=(15,7))

p1 = plt.bar(h_goals_year["Decade"],h_goals_year["Home_Number_goals"],color="g",width=5,label="Wins")

p2 = plt.bar(a_goals_year["Decade"],a_goals_year["Away_Number_goals"],bottom=h_goals_year["Home_Number_goals"],width=5,color="r",label="Loses")

plt.xticks(h_goals_year["Decade"])

plt.legend()
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,7))

sns.countplot(y = df["city"],order=df["city"].value_counts().index[:15],ax=ax1)

sns.countplot(y = df["country"],order=df["country"].value_counts().index[:15], ax=ax2)

fig.suptitle("Most frequent locations", fontsize=20)
fig,ax = plt.subplots(figsize=(15,7))

sns.countplot(y = df["tournament"],order=df["tournament"].value_counts().index[:15])

fig.suptitle("Most games per tournament", fontsize=20)
best_tournaments = df[df["tournament"].isin(df["tournament"].value_counts().index[:12])]

df_cups = best_tournaments.pivot_table(index=best_tournaments["date"],

                                columns=["tournament"],aggfunc="size", fill_value=0).resample("Y").sum()
#Test

plt.style.use('seaborn-darkgrid')

palette = plt.get_cmap('Set2')

num=0

ax,fix = plt.subplots(figsize=(15,7))

for column in df_cups:

    num+=1

    plt.subplot(3,4, num)

    for v in df_cups:

        plt.plot(df_cups.index,v,data=df_cups,marker='', color='black', linewidth=0.9, alpha=0.3)

        plt.tick_params(labelbottom=False)

        plt.plot(df_cups.index,column, data=df_cups,color="red", linewidth=2.4, alpha=0.9, label=column)

        plt.title(column, loc='left', fontsize=12, fontweight=0, color="black")

        plt.suptitle("Historic increase of tournament games ", fontsize=20, fontweight=0, color='black', style='italic', y=1.02)
Fifa_2018_teams = ["Argentina","Australia","Belgium","Brazil","Colombia","Costa Rica","Croatia","Denmark","Egypt","England","France","Germany","Iceland","Iran","Japan",

"South Korea","Mexico","Morocco","Nigeria","Panama","Peru","Poland","Portugal","Russia","Saudi Arabia","Senegal","Serbia","Spain","Sweden",

"Switzerland","Tunisia","Uruguay"]

df_fifa_teams = df[df["Winner"].isin(Fifa_2018_teams)]
fifa_teams_wins = df_fifa_teams.groupby(["Year","Winner"])["Winner"].agg("count")

fifa_teams_wins = pd.DataFrame(fifa_teams_wins)

fifa_teams_wins["Country"] = fifa_teams_wins.index.get_level_values(1)

fifa_teams_wins["Date"] = fifa_teams_wins.index.get_level_values(0)

#fifa_teams_lose = df_fifa_teams.groupby(["Year","Loser"])["Loser"].agg("count")

#fifa_teams_lose = pd.DataFrame(fifa_teams_lose)

#fifa_teams_lose["Country"] = fifa_teams_lose.index.get_level_values(1)

#fifa_teams_lose["Date"] = fifa_teams_lose.index.get_level_values(0)
g = sns.FacetGrid(fifa_teams_wins, col="Country", hue="Country", col_wrap=4)

g = g.map(plt.plot, "Date", "Winner").set_titles("{col_name}")
south_america = ["Brazil","Argentina","Uruguay","Colombia"]

europe = ["Germany","England","France","Spain"]
europe_wins = fifa_teams_wins[fifa_teams_wins["Country"].isin(europe)]
mean_europe = europe_wins.groupby("Year")["Winner"].agg(["mean"])
fig,[[ax1, ax2],[ax3, ax4]] = plt.subplots(2,2,figsize=(14,7),sharey=True)

#fig,ax = plt.subplots(2,2,figsize=(14,7))

ax1.plot("Date","Winner",data=europe_wins[europe_wins["Country"]=="England"],color="r",label="England wins")

ax1.plot(mean_europe.index,mean_europe["mean"],color = "black",label = "EU mean")

ax1.title.set_text("English wins")



ax2.plot("Date","Winner",data=europe_wins[europe_wins["Country"]=="France"],color="skyblue")

ax2.plot(mean_europe.index,mean_europe["mean"],color = "black",label = "EU mean")

ax2.title.set_text("French wins")



ax3.plot("Date","Winner",data=europe_wins[europe_wins["Country"]=="Germany"],color="y")

ax3.plot(mean_europe.index,mean_europe["mean"],color = "black",label = "EU mean")

ax3.title.set_text("German wins")



ax4.plot("Date","Winner",data=europe_wins[europe_wins["Country"]=="Spain"],color="orange")

ax4.plot(mean_europe.index,mean_europe["mean"],color = "black",label = "EU mean")

ax4.title.set_text("Spanish wins")
sa_wins = fifa_teams_wins[fifa_teams_wins["Country"].isin(south_america)]

sa_mean = sa_wins.groupby("Year")["Winner"].agg(["mean"])
fig,[[ax1, ax2],[ax3, ax4]] = plt.subplots(2,2,figsize=(14,7),sharey=True)



ax1.plot("Date","Winner",data=sa_wins[sa_wins["Country"]=="Brazil"],color="g")

ax1.plot(sa_mean.index,"mean",data=sa_mean,color="black")

ax1.title.set_text("Brazilian wins")



ax2.plot("Date","Winner",data=sa_wins[sa_wins["Country"]=="Argentina"],color="b")

ax2.plot(sa_mean.index,"mean",data=sa_mean,color="black")

ax2.title.set_text("Argentinian wins")



ax3.plot("Date","Winner",data=sa_wins[sa_wins["Country"]=="Colombia"],color="red")

ax3.plot(sa_mean.index,"mean",data=sa_mean,color="black")

ax3.title.set_text("Colombian wins")



ax4.plot("Date","Winner",data=sa_wins[sa_wins["Country"]=="Uruguay"],color="y")

ax4.plot(sa_mean.index,"mean",data=sa_mean,color="black")

ax4.title.set_text("Uruguay wins")
fig,ax =plt.subplots(figsize=(14,7))

ax.plot(sa_mean.index,"mean",data=sa_mean,color="orange",label="SA mean")

ax.plot(mean_europe.index,"mean",data=mean_europe,color="skyblue",label = "EU mean")

ax.title.set_text("European mean vs SA mean")

ax.legend()
europe_goals = df[df["home_team"].isin(europe)]

sa_goals = df[df["home_team"].isin(south_america)]
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

sns.distplot(europe_goals["home_score"] , color="skyblue", ax=ax1)

sns.distplot(sa_goals["home_score"]  , color="olive", ax=ax2)
f, (ax1,ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)

sns.distplot(europe_goals["away_score"] , color="skyblue", ax=ax1)

sns.distplot(sa_goals["away_score"]  , color="olive", ax=ax2)