# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import PercentFormatter 

raw = pd.read_csv("../input/ipl/deliveries.csv")

def renaming(df):
    return df.replace({
        "Deccan Chargers" : "Sunrisers Hyderabad",
        "Rising Pune Supergiants": "Chennai Super Kings",
        "Rising Pune Supergiant" : "Chennai Super Kings",
        "Gujarat Lions" : "Rajasthan Royals",
        "Pune Warriors" : "Others",
        "Kochi Tuskers Kerala" : "Others",
        "" : "Others"
        })

raw["batting_team"] = renaming(raw["batting_team"])
raw["bowling_team"] = renaming(raw["bowling_team"])

raw.head()

#1
sns.set()

batsman = raw.groupby("batsman")["batsman_runs"].sum().sort_values(ascending=False).head(15).reset_index()
plt.figure(figsize = (12,6))
sns.barplot("batsman","batsman_runs",data = batsman)
plt.xticks(rotation='vertical')
plt.xlabel("Batsman")
plt.ylabel("Total Runs Scored")
plt.title("Top 15 Batsman with highest runs scored")
plt.show()

#2
bowler = raw.groupby("bowler")["total_runs"].sum().sort_values(ascending=False).head(15).reset_index()
plt.figure(figsize = (12,6))
sns.barplot("bowler","total_runs",data = bowler)
plt.xticks(rotation='vertical')
plt.xlabel("Bowler")
plt.ylabel("Total Runs Given")
plt.title("Top 15 Bowlers who gave highest runs")
plt.show()

#3
    
w_types = raw["dismissal_kind"].unique()
w = [1,2,8] + list(range(4,7))
w_types = [w_types[x] for x in w]

temp = raw.loc[raw["dismissal_kind"].isin(w_types)].reset_index(drop=True).groupby(['bowler','dismissal_kind']).dismissal_kind.count().unstack(level=1, fill_value = 0)
temp["total"] = temp.sum(axis =1)
runsgiven = temp.sort_values("total", ascending = False).head(10).drop("total", axis = 1)
runsgiven.columns = map(lambda x:x.upper(), list(runsgiven.columns))

runsgiven.plot(kind = "bar", figsize = (12,6), stacked = True)
plt.xlabel("Bowler")
plt.ylabel("Total Wickets Taken")
plt.legend()
plt.title("Top 10 Highest Wicket-Takers categorized by Type of Wicket Taken")


#4 

team_avg = raw.groupby(["batting_team","match_id"])["total_runs"].aggregate("sum").reset_index()\
.groupby("batting_team")["total_runs"].agg(average="mean", n="count")\
.reset_index().sort_values("average", ascending = False).reset_index(drop = True)

plt.figure(figsize = (12,6))
sns.pointplot("batting_team","average","batting_team", data = team_avg)
plt.xticks(rotation = "vertical")
plt.legend("")
plt.ylabel("Average Score")
plt.xlabel("Batting Team")
plt.title("Averags Scores of each Team")
plt.ylim(135,165)
plt.show()


# 5

sr = lambda x: x.mean()*100

temp = raw.groupby("batsman")["batsman_runs"].agg(rate=sr,n="count").reset_index()
strike = temp[temp["n"]>500].sort_values("rate", ascending = False).reset_index(drop = True)[:10]

plt.figure(figsize = (12,6))
sns.pointplot("batsman","rate","batsman", data = strike)
plt.xticks(rotation = "vertical")
plt.legend("")
plt.ylabel("Strike Rate")
plt.xlabel("Batsman")
plt.title("Batsmen with Highest S/R (Balls Played > 500)")
plt.ylim(125,175)
plt.show()

#6

df1 = raw.groupby(["match_id","inning","batting_team"])["total_runs"].sum().reset_index().pivot("match_id","inning","batting_team").reset_index(drop = True)
df2 = raw.groupby(["match_id","inning","batting_team"])["total_runs"].sum().reset_index().pivot("match_id","inning","total_runs").reset_index(drop = True)

#Matches with super overs
df2.loc[df2[1]==df2[2]]

#Removing matches with super overs then joining
df1 = df1.loc[df2[1]!=df2[2]][[1,2]]
df2 = df2.loc[df2[1]!=df2[2]][[1,2]]

df1.columns = ["t1","t2"]
match = pd.concat([df1, df2], axis=1, join='inner')
match["Won"] = np.where(match[1]>match[2], match["t1"], match["t2"])
match["Lost"] = np.where(match[1]>match[2], match["t2"], match["t1"])
match.reset_index(inplace = True)


df = pd.melt(match, value_vars=['Won', 'Lost'], id_vars = "index").groupby(["variable","value"]).count().reset_index().sort_values(["variable","index"], ascending = False)

plt.figure(figsize = (12,6))
sns.barplot("value","index","variable", data = df)
plt.xticks(rotation = "vertical")
plt.xlabel("Teams")
plt.ylabel("No. of Wins(blue) and Losses(orange)")
plt.title("No. of Wins and Loses Per Team")

#7
    
batsman = raw.groupby(["batting_team","batsman"])["batsman_runs"].agg("sum").reset_index().sort_values(["batting_team","batsman_runs"], ascending = False).groupby("batting_team").head(1).sort_values("batsman_runs", ascending = False).reset_index(drop = True)
batsman["team_player"] = batsman["batting_team"] + "\n" + batsman["batsman"]

plt.figure(figsize=(12,6))
sns.barplot("team_player","batsman_runs", data=batsman.loc[batsman["batting_team"]!="Others",:])
plt.xticks(rotation = "vertical")
plt.ylabel("Runs Scored")
plt.xlabel("Team and Best Batsman")
plt.title("Getting Best Batsman of all the Teams")



#8

bowler = raw.loc[raw["dismissal_kind"].isin(w_types)].reset_index(drop=True).groupby(['bowling_team','bowler']).dismissal_kind.count().reset_index().sort_values(["bowling_team","dismissal_kind"], ascending = False).groupby("bowling_team").head(1).sort_values("dismissal_kind", ascending = False).reset_index(drop = True)
bowler["team_player"] = bowler["bowling_team"] + "\n" + bowler["bowler"]

plt.figure(figsize=(12,6))
sns.barplot("team_player","dismissal_kind", data=bowler.loc[bowler["bowling_team"]!="Others",:])
plt.xticks(rotation = "vertical")
plt.ylabel("Wickets Take")
plt.xlabel("Team and Best Bowler")
plt.title("Getting Best Bowlers of all the Teams")

#9
#Bowled most overs and Batted most balls

bowler = raw.groupby(["bowler","total_runs"])["total_runs"].agg(n = "count").unstack(fill_value = 0)    
bowler["total"] = np.floor(bowler.sum(axis=1)/6)
bowler = bowler.sort_values("total",ascending = False)[:12].drop("total", axis = 1)
bowler.columns = list(map(str,(list(range(0,8)))))
bowler.plot(kind = "bar", figsize = (12,6), stacked = True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title = "Runs Given")
plt.title("Bowlers with Highest # of Balls and Runs composition of total balls bowled")


batsman = raw.groupby("batsman")["batsman"].agg(n = "count").reset_index().sort_values("n",ascending = False)[:12]
plt.figure(figsize = (12,6))
sns.barplot("batsman","n", data = batsman)
plt.xticks(rotation = "vertical")
plt.ylabel("No. of balls played")
plt.title("Batsman with most number of balls played")

plt.show()



#10

np.random.seed(22)
prop= pd.melt(match, value_vars=['Won', 'Lost'], id_vars = "index").groupby("value").apply(lambda x:x.sample(25)).reset_index(drop = True).groupby(["variable","value"]).count().reset_index().sort_values(["variable","index"], ascending = False).pivot("value","variable").reset_index()
prop["total"] = prop["index"].sum(axis=1)
prop["prob"] = prop["index"]["Won"]/prop["total"]
prop["up"] = prop["prob"] + 1.96*np.sqrt(prop["prob"]*(1-prop["prob"])/prop["total"])
prop["down"] = prop["prob"] - 1.96*np.sqrt(prop["prob"]*(1-prop["prob"])/prop["total"])
prop.sort_values("prob", inplace=True)

plt.figure(figsize = (12,6))
sns.pointplot("value","prob", hue = "value", data=prop, join = False)
sns.pointplot("value","up", color = "black", data=prop, join = False)
sns.pointplot("value","down", color = "black", data=prop, join = False)
plt.legend("")
plt.xlabel("Team")
plt.ylabel("Win Percentage")
plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
plt.xticks(rotation = "vertical")
plt.title("Winning Percentages of all the teams with Sample Confidence Intervals")

plt.show()

