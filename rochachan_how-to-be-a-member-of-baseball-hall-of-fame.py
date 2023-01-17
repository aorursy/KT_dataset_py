%matplotlib inline

%config InlineBackend.figure_format = 'retina'

import os

from sklearn import preprocessing

import numpy as np

import pandas as pd

import seaborn as sns

from wordcloud import WordCloud

from matplotlib import pyplot as plt

pd.set_option("display.max_columns",50)
### read in data

hall_of_fame = pd.read_csv("../input/hall_of_fame.csv")

## to choose only the player 

hall_of_fame = hall_of_fame[hall_of_fame.category == "Player"]

print(hall_of_fame.info())

hall_of_fame.head()
## calculate votedby data with inducted 

voted_by = hall_of_fame.groupby(["votedby","inducted"]).size().unstack()

norm_voted_by = pd.DataFrame(preprocessing.normalize(voted_by.fillna(0)),columns=voted_by.columns,index=voted_by.index)

norm_voted_by["delta"] = norm_voted_by["Y"] - norm_voted_by["N"]



##plot

fig,axes_voted = plt.subplots(2,1,figsize=(12,16))

colors = ["r","r","r","r","b","b","b","b"]

voted_by[["N","Y"]].plot.barh(title="Whose vote is most powerful 1 ?",ax=axes_voted[0])

norm_voted_by.delta.sort_values().plot.barh(title="Whose vote is most powerful 2 ?",\

                                            ax=axes_voted[1],color=colors)

hall_of_fame[hall_of_fame.inducted == "Y"].groupby("yearid").size().describe()
## calculate the year with inducted

induct = hall_of_fame.groupby(["yearid","inducted"]).size().unstack().fillna(0)

induct["inducted rate"] = induct["Y"]/(induct["Y"]+induct["N"])



fig,axes_induct = plt.subplots(1,1,figsize=(16,10))

ax1 = induct.plot(y="inducted rate",kind="line",style="ro-",secondary_y=True,use_index=False,ax=axes_induct)

induct["Y"].plot(kind = "bar",title="number of people chosen into the HOF each year"\

                 ,ax=axes_induct,label="HOF",legend=True)
hall_of_fame.groupby("player_id").size().describe()
hall_of_fame.groupby("player_id").size().hist()
hall_of_fame[["yearid","needed"]].groupby("yearid").mean().fillna(0).plot.bar(figsize=(18,8),style = "o-",title="vote needed")
## read in player table

player = pd.read_csv("../input/player.csv",parse_dates=["debut","final_game"])

player["serviceYear"] = player["final_game"] - player["debut"]

# player["serviceYear"] = player.serviceYear.astype('timedelta64[D]')



## label the player whether they're enter into HOF

player = player.join(hall_of_fame[hall_of_fame.inducted == "Y"][["player_id","inducted"]].set_index("player_id"),\

                     on="player_id")

player.inducted.fillna("N",inplace=True)



print(player.info())

player.head()
wordcloud=WordCloud().generate_from_frequencies(player.name_first.value_counts().to_dict())

plt.figure(figsize=(10,8))

plt.imshow(wordcloud, interpolation="bilinear")

plt.axis("off")

plt.show()
player.describe()
player[player["serviceYear"] < pd.Timedelta("1 days")].head()
player[player["serviceYear"] < pd.Timedelta("0 days")]
player.loc[player["serviceYear"].argmax()]
player[player.inducted == "Y"].serviceYear.astype('timedelta64[Y]').describe()
player[player.inducted == "Y"][player[player.inducted == "Y"].serviceYear.astype('timedelta64[Y]')<1]
sns.distplot(player.serviceYear.astype('timedelta64[Y]').dropna(),\

            kde_kws={"lw":3,"label":"Total People"})

g = sns.distplot(player[player.inducted == "Y"].serviceYear.astype('timedelta64[Y]').dropna(),\

             kde_kws={"lw": 3, "label": "Hall Of Fame"})

g.set_title("how many year takes player enter into the hall of fame ?")
fig,axes_w_h = plt.subplots(1,2,figsize=(18,6))

sns.distplot(player["height"].dropna(),ax=axes_w_h[0],hist_kws={"label":"Player's height histplot"})

axes_w_h[0].set_title("Player's height histplot")

axes_w_h[0].legend()

sns.distplot(player["weight"].dropna(),ax=axes_w_h[1],hist_kws={"label":"Player's weight histplot"})

axes_w_h[1].set_title("Player's weight histplot")

axes_w_h[1].legend()
g = sns.FacetGrid(player, hue="inducted", size=7,aspect=1.5,palette=sns.color_palette(sns.color_palette("Paired")))

g.map(plt.scatter, "height", "weight", s=50, edgecolor="white")

plt.title("What's the player admitted into the hall of fame?")

plt.legend()
batting = pd.read_csv("../input/batting.csv")

print(batting.info())

batting.head()
player_batting = batting.groupby("player_id").sum().iloc[:,2:].fillna(0)

player_batting["ba"] = player_batting["h"].div(player_batting["ab"],fill_value=0)

player_batting = player_batting.join(player[["player_id","inducted"]].set_index("player_id"))
g = sns.pairplot(player_batting,vars=["g","h","double","triple","hr"],hue="inducted")

plt.title("batting statics pairplot")
plt.figure(figsize=(14,10))

g = sns.heatmap(player_batting.corr(),vmin=0,vmax=1,linewidths=.5,cmap="OrRd",annot=True)

g.set_title("batting statistics correlation heatmap",fontdict={"fontsize":15})
sns.factorplot(x="variable",y="value",hue="inducted",\

               data = pd.melt(player_batting[["ab","r","h","inducted"]],id_vars="inducted"),\

               kind="box",size=10,aspect=1.5,showfliers=False)

plt.title("Compare boxplot 1")
sns.factorplot(x="variable",y="value",hue="inducted",data = pd.melt(player_batting.iloc[:,4:],id_vars="inducted")\

               ,kind="box",size=10,aspect=1.5,showfliers=False)

plt.title("Compare boxplot 2")
### batting average

sns.distplot(player_batting.ba.dropna(),label= "Normal player")

g = sns.distplot(player_batting[player_batting["inducted"] == "Y"].ba.dropna(),label= "Hall Of Fame")

g.set_title("Batting Average distribution")

plt.legend()
salary = pd.read_csv("../input/salary.csv")

### join salary

player = player.join(salary.groupby(["player_id"])[["player_id","salary"]].mean(),on="player_id")

print(salary.info())

salary.head()
salary.player_id.describe()
g = salary.groupby(["year"]).salary.median().plot.bar(title="salary boom")

g.set_ylabel("salary")
player.salary.describe()
fig,axes_salary = plt.subplots(1,2,figsize=(20,8))

sns.distplot(player[player["inducted"] == "N"].salary.dropna(),label= "Normal player",ax=axes_salary[0])

sns.distplot(player[player["inducted"] == "Y"].salary.dropna(),label= "Hall Of Fame",ax=axes_salary[0])

g = sns.boxplot(x="inducted",y="salary",data=player,ax=axes_salary[1])

axes_salary[0].legend()



### read in award data

awards = pd.read_csv("../input/player_award.csv")

### label player with inducted data

awards = awards.join(player[["player_id","inducted"]].set_index("player_id"),on="player_id")

award_count = awards.groupby("player_id").size()

award_count.name = "award_count"

### label the number of awards to payer table

player = player.join(award_count,on="player_id")

print(awards.info())

awards.head()
awards.player_id.describe()
awards[awards["player_id"] == "bondsba01"]
awards.groupby("player_id").size().describe()
player.award_count.plot.hist()
awarded = awards.groupby(["award_id","inducted"]).size().unstack()

awarded = awarded.fillna(0)

awarded["delta"] = awarded["Y"] - awarded["N"]

awarded["ratio"] = awarded["Y"]/awarded["N"]



fig,axes_award = plt.subplots(2,1,figsize=(14,20))

awarded.sort_values(by="delta",ascending=True)[["N","Y"]].plot(kind="barh",ax=axes_award[0]\

                                                               ,title="Pass of Awarded into HOF")

awarded.ratio.sort_values().plot.barh(ax=axes_award[1],title = "Admited into HOF ratio compare")
sns.boxplot(x="inducted",y="award_count",data=player)