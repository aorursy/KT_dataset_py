import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import itertools

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

winners = df[df.id.isin(df.groupby("season").id.max())]

winners[["season", "winner"]]
deliveries = pd.read_csv("../input/deliveries.csv")

df

deliveries = deliveries.merge(df[["season", "id"]], left_by = "id", right_by ="match_id")

df.groupby(["season", "match_id"]).total_runs.sum()
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import itertools

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df["batting_team"] = np.where((df["batting_team"] == "Rising Pune Supergiant"), "Rising Pune Supergiants", df["batting_team"])

df["wicket"] = np.where(((df.player_dismissed.notnull())&(df.dismissal_kind != "retired hurt")&(df.dismissal_kind != "run out")), 1, 0)



def func(x):

    y = x.groupby("bowler").wicket.agg({"wicket":"sum"}).reset_index()

    z = list(map(list, list(itertools.combinations(y["bowler"], 2))))

    d = pd.DataFrame(data={"A":z})

    d["bowler"] = d.A.map(lambda x: x[0])

    d["bowler2"] = d.A.map(lambda x: x[1])

    d.drop(columns = "A", inplace = True)

    d = pd.merge(d, y, on="bowler", how = "outer")

    d = d.rename(columns={"bowler": "bowler1", "bowler2": "bowler", "wicket":"wickets"})

    d = pd.merge(d, y, on="bowler", how = "outer")

    d["total_wickets"] = d["wicket"]+d["wickets"]

    d = d.rename(columns={"bowler": "bowler1", "bowler": "bowler2"})

    d = d.drop(columns = ["wicket", "wickets"])

    d = d.dropna()

    #d = d.rename(columns={})

    return d

    #print(d)

bowler_wickets = df.groupby(["match_id", "inning", "bowling_team"])["bowler", "wicket"].apply(lambda x: func(x)).reset_index().drop(columns = "level_3")

bowler_wickets[["bowler1", "bowler2"]] = np.sort(bowler_wickets[["bowler1", "bowler2"]], 1)

res = bowler_wickets.groupby(["bowler1", "bowler2"]).total_wickets.sum().reset_index().nlargest(10, "total_wickets")#.rename(columns={"bowler": "bowler1", "bowler2": "bowler", "wicket":"wickets"})

res["bowlers"] = res["bowler1"].astype(str) + " - " + res["bowler2"].astype(str)

#res = res.drop(columns = ["bowler1", "bowler2"])

res

fig = plt.figure()

ax = plt.subplot(111)

ax.bar(res["bowlers"], res["total_wickets"], width=0.5)

xlocs, xlabs = plt.xticks()

xlocs= np.arange(0,10)

plt.xticks(res["bowlers"], res["bowlers"], rotation='vertical')

for i, v in enumerate(res["total_wickets"]):

    plt.text(xlocs[i] - 0.25, v, str(v))

plt.show()
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import itertools

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

mapper = df.groupby(['match_id', 'inning']).batsman.apply(lambda x: dict(zip(x[~x.duplicated()], np.arange(1, len(x[~x.duplicated()])+1)))).reset_index(name = 'batting_position').rename(columns = {'level_2':'batsman'})

df = df.merge(mapper, on = ['match_id', 'inning', 'batsman'], how = 'outer')

df["batting_position"] = df["batting_position"].replace(2, 1)

df = df[df.is_super_over == 0]

df["wicket"] = np.where(((df.player_dismissed.notnull())&(df.dismissal_kind != "retired hurt")&(df.dismissal_kind != "run out")), 1, 0)

df["wicket_value"] = np.where(((df["batting_position"] <= 3)&(df["wicket"] == 1)), 1, 0)

df["wicket_value"] = np.where(((df["batting_position"] >= 4)&(df["batting_position"] <= 7)&(df["wicket"] == 1)), 0.66, df["wicket_value"])

df["wicket_value"] = np.where(((df["batting_position"] > 8)&(df["wicket"] == 1)), 0.33, df["wicket_value"])



def func(x):

    bowler_with_wickets = x.groupby("bowler")["wicket", "wicket_value"].sum().reset_index()

    z = list(map(list, list(itertools.combinations(bowler_with_wickets["bowler"], 2))))

    d = pd.DataFrame(data={"bowler1": np.array(z)[:,0], "bowler2": np.array(z)[:,1]})

    d = d.merge(bowler_with_wickets, left_on="bowler1", right_on = "bowler", how = "outer").drop(columns = "bowler").rename(columns = {"wicket": "bowler1_wickets", "wicket_value":"bowler1_value"}).merge(bowler_with_wickets, left_on="bowler2", right_on = "bowler", how = "outer").drop(columns = "bowler").rename(columns = {"wicket": "bowler2_wickets", "wicket_value":"bowler2_value"}).dropna()

    d["total_wickets"] = d["bowler1_wickets"]+d["bowler2_wickets"]

    d["total_value"] = d["bowler1_value"]+d["bowler2_value"]    

    return d

    

bowler_wickets = df.groupby(["match_id", "inning"])["bowler", "wicket", "wicket_value"].apply(lambda x: func(x)).reset_index()#.drop(columns = "level_2")

bowler_wickets[["bowler1", "bowler2"]] = np.sort(bowler_wickets[["bowler1", "bowler2"]], 1)

res = bowler_wickets.groupby(["bowler1", "bowler2"])["total_wickets", "total_value"].sum().reset_index().nlargest(10, "total_value")#.rename(columns={"bowler": "bowler1", "bowler2": "bowler", "wicket":"wickets"}) 

fig = plt.figure()

ax = plt.subplot(111)

ax.bar(res["bowler1"]+" - "+res["bowler2"], res["total_wickets"], width=0.5)

xlocs, xlabs = plt.xticks()

xlocs= np.arange(0,10)

plt.xticks(res["bowler1"]+" - "+res["bowler2"], res["bowler1"]+" - "+res["bowler2"], rotation='vertical')

for i, v in enumerate(res["total_wickets"]):

    plt.text(xlocs[i] - 0.25, v, str(v))

plt.show()
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df_matches = pd.read_csv("../input/matches.csv")

df_matches = df_matches.rename(columns = {"id":"match_id"})

df = pd.merge(df, df_matches[["match_id", "season"]], on="match_id")

def func(x):

    players = pd.unique(np.concatenate((x.batsman.unique(), x.bowler.unique(), x.fielder.unique()), axis = 0))

    runs_scored = x.groupby("batsman").batsman_runs.agg({"runs_scored": "sum"}).reset_index().rename(columns = {"batsman":"player"})

    x["total_runs"] = x["total_runs"] - x["bye_runs"] - x["legbye_runs"] - x["penalty_runs"]

    runs_given = x.groupby("bowler").total_runs.agg({"runs_given":"sum"}).reset_index().rename(columns = {"bowler":"player"})

    fielding = x.groupby("fielder").fielder.agg({"fielding":"count"}).reset_index().rename(columns = {"fielder":"player"})

    x = x[(x["dismissal_kind"] != "run out") & (x["dismissal_kind"] != "obstructing the field") & (x["dismissal_kind"] != "retired hurt")]

    wickets = x.groupby("bowler").player_dismissed.agg({"wickets":"count"}).reset_index().rename(columns = {"bowler":"player"})

    res = pd.DataFrame(data = {"player":players}, index = np.arange(1, len(players)+1))

    res = res.dropna()

    res = pd.merge(res, runs_scored, on="player", how = "outer")

    res = pd.merge(res, runs_given, on="player", how = "outer")

    res = pd.merge(res, fielding, on="player", how = "outer")

    res = pd.merge(res, wickets, on="player", how = "outer")

    res = res.fillna(0)

    return res

    

impact = df.groupby("match_id").apply(lambda x: func(x)).reset_index().drop(columns = "level_1")

impact["points"] = (impact["runs_scored"]/30 + impact["wickets"] - impact["runs_given"]/20 + impact["fielding"]/2)

impact.nlargest(20, "points")
import pandas as pd

import numpy as np

import warnings



warnings.simplefilter(action = "ignore", category = FutureWarning)



deliveries = pd.read_csv("../input/deliveries.csv")

all_players = pd.unique(np.concatenate((deliveries["batsman"].unique(), deliveries["bowler"].unique(), deliveries["fielder"].unique()), axis = 0))

players = pd.DataFrame(data = {"players": all_players.tolist()}, index = np.arange(1, len(all_players.tolist())+1))

batters = deliveries.groupby("batsman").batsman_runs.sum().reset_index().rename(columns = {"batsman":"players", "batsman_runs":"Runs"})

fielders = deliveries.groupby("fielder").dismissal_kind.agg("count").reset_index().rename(columns = {"fielder":"players", "dismissal_kind":"fielding"})

deliveries = deliveries[(deliveries["dismissal_kind"] != "run out") & (deliveries["dismissal_kind"] != "obstructing the field") & (deliveries["dismissal_kind"] != "retired hurt")]

bowlers = deliveries.groupby("bowler")["player_dismissed"].agg({"wickets":"count"}).reset_index().rename(columns = {"bowler":"players"})

players = pd.merge(players, batters, on = "players", how = "outer")

players = pd.merge(players, fielders, on = "players", how = "outer")

players = pd.merge(players, bowlers, on = "players", how = "outer").fillna(0)

players["points"] = np.round(players["Runs"]/30 + players["wickets"] + players["fielding"], 2)

players.nlargest(10,"points")
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df = df_matches.rename(columns = {"id":"match_id"})



def func(x):

    return x.sort_values(by = "matches", ascending=False)#.nlargest(3, "matches")



z = df.groupby(["winner", "venue"])["match_id"].count().reset_index().rename(columns = {"winner":"team", "match_id":"matches"})

z = z.groupby("team").apply(lambda x: func(x)).drop(["team"], axis = 1).reset_index().drop(columns = ["level_1"])



def func1(x):

    teams = np.concatenate((x.team1.values, x.team2.values), axis = 0)

    return pd.Series(teams).value_counts().reset_index()



y = df.groupby(["venue"])["match_id", "team1", "team2"].apply(lambda x: func1(x)).reset_index().drop(columns = ["level_1"]).rename(columns = {"index":"team", 0:"played"})

y = y.groupby("team")["venue", "played"].apply(lambda x: x.sort_values(by = "played", ascending=False)).reset_index().drop(columns = ["level_1"])





mgrd_df = pd.merge(z, y, on = ["team", "venue"], how = "outer")

mgrd_df.fillna(0, inplace = True)

mgrd_df["win_pct"] = np.round(mgrd_df["matches"]/mgrd_df["played"]*100, 2)

mgrd_df = mgrd_df.sort_values(by = ["team", "matches"], ascending = [True, False])

#mgrd_df.to_csv("temp.csv",  sep=',', encoding='utf-8')

mgrd_df[mgrd_df["played"] > 10].sort_values(by = "win_pct", ascending = False)
from matplotlib import pyplot as plt

import pandas as pd

import seaborn as sns



df = pd.read_csv("../input/matches.csv")

plt.subplots(figsize = (10,6))

sns.countplot(x="season", hue = "toss_decision", data = df)

df.groupby("season")["toss_decision"].value_counts().unstack()
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

warnings.simplefilter(action = "ignore", category = FutureWarning)



deliveries = pd.read_csv("../input/deliveries.csv")

mapper = deliveries.groupby(['match_id', 'inning']).batsman.apply(lambda x: dict(zip(x[~x.duplicated()], np.arange(1, len(x[~x.duplicated()])+1)))).reset_index(name = 'batting_position').rename(columns = {'level_2':'batsman'})

deliveries_position = deliveries.merge(mapper, on = ['match_id', 'inning', 'batsman'], how = 'outer')

deliveries_position["batting_position"] = deliveries_position["batting_position"].replace(2, 1)

temp = deliveries_position.groupby(["batting_position", "batsman"])["batsman_runs"].agg("sum")

result = temp.groupby(level=0).nlargest(3).reset_index(level=0, drop=True).reset_index()

result
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

plt.subplots(figsize = (10,6))

umps = np.concatenate((df["umpire1"].values, df["umpire2"].values), axis = 0)

res = pd.Series(umps).value_counts().reset_index().rename(columns = {"index": "Umpire", 0: "Matches"})

res.nlargest(10, "Matches")
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import warnings



warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df["venue"] = np.where((df["venue"] == "Punjab Cricket Association IS Bindra Stadium, Mohali"), "Punjab Cricket Association Stadium, Mohali", df["venue"])

grounds = df["venue"].value_counts().reset_index().rename(columns = {"index": "venues", "venue": "Matches"})

grounds.nlargest(10, "Matches")
import pandas as pd

import numpy as np

import warnings



warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df["venue"] = np.where((df["venue"] == "Punjab Cricket Association IS Bindra Stadium, Mohali"), "Punjab Cricket Association Stadium, Mohali", df["venue"])

city = df["city"].value_counts().reset_index().rename(columns = {"index": "city", "city": "matches"})

city.nlargest(10, "matches")
import pandas as pd

import numpy as np

import warnings



warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df["venue"] = np.where((df["venue"] == "Punjab Cricket Association IS Bindra Stadium, Mohali"), "Punjab Cricket Association Stadium, Mohali", df["venue"])

#res = df[["city", "venue"]].groupby(["city"]).venue.nunique().reset_index().sort_values(by = "venue", ascending = False)

#res[res.venue>1]

def func(x):

    if(len(x.venue.unique()) > 1):

        return pd.DataFrame(data = {"venues": x.venue.unique()}, index = np.arange(1, len(x.venue.unique())+1))

    

res = df[["city", "venue"]].groupby(["city"]).apply(lambda x: func(x)).reset_index(level = 1, drop = True)

res
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



deliveries = pd.read_csv("../input/deliveries.csv")

def func1(x):

    All = pd.DataFrame(data = {"Player": pd.unique(np.concatenate((x.batsman.unique(), x.bowler.unique(), x.fielder.unique())))})

    return All



mapper = deliveries.groupby('match_id')["batsman", "bowler", "fielder"].apply(lambda x: func1(x)).reset_index().drop(["level_1"], axis = 1).dropna()

mapper.groupby("Player").match_id.count().reset_index().rename(columns = {"match_id":"matches"}).nlargest(10,"matches")
import pandas as pd

import numpy as np

import warnings



warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df.groupby("player_of_match").id.agg("count").reset_index().rename(columns = {"id":"mom"}).nlargest(10, "mom")

import pandas as pd

import numpy as np

import warnings



warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df.groupby(["player_of_match", "season"]).id.agg("count").reset_index().rename(columns = {"id":"mom"}).nlargest(10, "mom")

import pandas as pd

import numpy as np

import warnings



warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df.groupby("season").apply(lambda x: x.player_of_match.value_counts().nlargest(1)).reset_index().rename(columns = {"level_1":"player"})
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df1 = pd.read_csv("../input/matches.csv")

df = pd.read_csv("../input/deliveries.csv")

df1 = df_matches.rename(columns = {"id":"match_id"})

df = pd.merge(df, df1[["match_id", "season"]], on = "match_id")

df = df[(df["is_super_over"] != 1)]

def func(x):

    a = x.total_runs.sum()/(x.match_id.nunique()*2)

    b = x.total_runs.where(x.inning == 1).sum()/(x.match_id.nunique())

    c = x.total_runs.where(x.inning == 2).sum()/(x.match_id.nunique())

    res = pd.DataFrame(data = {"Match":a, "1st innings":b, "2nd innings":c}, index = [1])

    return res

result = df.groupby("season")["inning", "total_runs", "match_id"].apply(lambda x: func(x)).reset_index().drop(columns = "level_1")

result
import pandas as pd

import matplotlib.pyplot as plt

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



matches = pd.read_csv("../input/matches.csv")

deliveries = pd.read_csv("../input/deliveries.csv")

matches = matches.rename(columns = {"id":"match_id"})

deliveries = pd.merge(deliveries, matches[["match_id", "season"]], on = "match_id")

deliveries = deliveries[(deliveries["is_super_over"] != 1)]

deliveries = deliveries[["match_id", "batting_team" ,"season", "inning", "over", "ball", "total_runs", "player_dismissed"]]

powerplay = deliveries[(deliveries.over <= 6)]

middleovers = deliveries[(deliveries.over >= 7) & (deliveries.over <= 14)]

slogovers = deliveries[(deliveries.over >= 15)]

a = powerplay.groupby(["season", "batting_team"])["total_runs", "match_id", "player_dismissed"].apply(lambda x: pd.DataFrame(data = {"Runrate(1-6)" : np.round(x.total_runs.sum()/(x.match_id.nunique()*6),2), "wickets(1-6)":np.round(x.player_dismissed.dropna().count()/(x.match_id.nunique()), 0)}, index = np.arange(1))).reset_index().drop("level_2", axis = 1)

b = middleovers.groupby(["season", "batting_team"])["total_runs", "match_id", "player_dismissed"].apply(lambda x: pd.DataFrame(data = {"Runrate(7-14)" : np.round(x.total_runs.sum()/(x.match_id.nunique()*8),2), "wickets(7-14)":np.round(x.player_dismissed.dropna().count()/(x.match_id.nunique()), 0)}, index = np.arange(1))).reset_index().drop("level_2", axis = 1)

c = slogovers.groupby(["season", "batting_team"])["total_runs", "match_id", "player_dismissed"].apply(lambda x: pd.DataFrame(data = {"Runrate(15-20)" : np.round(x.total_runs.sum()/(x.match_id.nunique()*6),2), "wickets(15-20)":np.round(x.player_dismissed.dropna().count()/(x.match_id.nunique()), 0)}, index = np.arange(1))).reset_index().drop("level_2", axis = 1)

report = pd.merge(a, b, on = (["season", "batting_team"]))

report = pd.merge(report, c, on = (["season", "batting_team"]))

report.groupby("season").apply(lambda x: x.sort_values(by = "wickets(7-14)", ascending=False)[:5]).drop(columns="season")
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

df = df.rename(columns = {"id":"match_id"})

z = df.groupby("winner")["toss_winner", "winner"].apply(lambda x: np.where(x.toss_winner == x.winner, 1, 0).sum()).reset_index().rename(columns = {"winner":"team", 0:"toss and match wins"})

y = df.groupby("toss_winner").apply(lambda x: x.toss_winner.count()).reset_index().rename(columns = {"toss_winner":"team", 0:"toss wins"})

team1 = df.groupby("team1").match_id.count().reset_index().rename(columns = {"team1":"team", "match_id":"a"})

team2 = df.groupby("team2").match_id.count().reset_index().rename(columns = {"team2":"team", "match_id":"b"})

matches_played = pd.merge(team1, team2, on = "team")

matches_played["played"] = matches_played["a"] + matches_played["b"]

matches_played.drop(columns = ["a", "b"], inplace = True)

z = pd.merge(matches_played, z, on = "team")

z = pd.merge(z, y, on = "team")

z["toss_win_pct"] = (z["toss wins"]/z["played"])*100

z["toss_win_con"] = (z["toss and match wins"]/z["toss wins"])*100

z[z.played>50].sort_values(by = ["toss_win_con", "played"], ascending=False)
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/matches.csv")

def func(x):

    matches_played = x.groupby("team1").id.agg({"match":"count"})["match"] + x.groupby("team2").id.agg({"match":"count"})["match"]

    matches_played = matches_played.reset_index().rename(columns = {"team1":"team"})

    won_batting_first = x.groupby("winner").apply(lambda x: np.count_nonzero(x.win_by_runs)).reset_index().rename(columns = {"winner":"team", 0:"won batting 1st"})

    won_chasing = x.groupby("winner").apply(lambda x: np.count_nonzero(x.win_by_wickets)).reset_index().rename(columns = {"winner":"team", 0:"won chasing"})

    res = pd.merge(matches_played, won_batting_first, on = "team")

    res = pd.merge(res, won_chasing, on = "team")

    res["wins"] = res["won batting 1st"] + res["won chasing"] 

    return res.sort_values(by = "wins", ascending = False)[:4]

df.groupby("season").apply(lambda x: func(x))
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

z = df.groupby("batsman")["batsman_runs"].apply(lambda x: x.value_counts()).unstack().reset_index().fillna(0)

z["Runs"] = z[1] + z[2]*2 + z[3]*3 + z[4]*4 + z[5]*5 + z[6]*6

z["Balls"] = z[[0,1,2,3,4,5,6]].sum(1)

z["bndry_pct"] = np.round((z[4]*4 + z[6]*6)/z["Runs"]*100, 2)

z["1s-2s_pct"] = np.round((z[1]+z[2]*2)/z["Runs"]*100, 2)

z["dots_pct"] = np.round(z[0]/z["Runs"]*100, 2)

z.fillna(0, inplace=True)

#z.sort_values(by = ["Runs", "dots_pct"], ascending = [False, False])

#z.nlargest(20, "Runs").nlargest(20, "dots_pct")

z.nlargest(10, "Runs")#.nlargest(20, "bndry_pct")
import pandas as pd

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

d1 = df.groupby(["batsman"])["batsman_runs"].agg({"runs":"sum"}).reset_index()

d2 = df.groupby("player_dismissed").player_dismissed.agg({"out": "count"}).reset_index().rename(columns = {"player_dismissed":"batsman"})

d3 = df.groupby(["batsman"])["batsman"].agg({"balls":"count"}).reset_index()

df2 = pd.merge(d1, d2, on = "batsman", how = "outer").fillna(0)

df2 = pd.merge(df2, d3, on = "batsman", how = "outer").fillna(0)

df2 = df2[df2.runs >= 2000] #minimum 2000 runs

df2["Average"] = (df2.runs/df2.out)

df2["Strike_Rate"] = (df2.runs/df2.balls)*100

df2[["batsman", "Average", "Strike_Rate"]].nlargest(10, "Average")
import numpy as np

import pandas as pd

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



deliveries = pd.read_csv("../input/deliveries.csv")

mapper = deliveries.groupby(['match_id', 'inning']).batsman.apply(lambda x: dict(zip(x[~x.duplicated()], np.arange(1, len(x[~x.duplicated()])+1)))).reset_index(name = 'batting_position').rename(columns = {'level_2':'batsman'})

deliveries_position = deliveries.merge(mapper, on = ['match_id', 'inning', 'batsman'], how = 'outer')

deliveries_position = deliveries_position[(deliveries_position.batting_position == 8) |(deliveries_position.batting_position == 9)

                                         |(deliveries_position.batting_position == 10) | (deliveries_position.batting_position == 11)]

asBatsman = deliveries_position.groupby(["batsman", "batting_position"]).apply(lambda x: pd.DataFrame(data = {"Runs": x.batsman_runs.sum(), "Matches played":x.match_id.nunique()}, index = [1])).reset_index().drop(columns = "level_2")

asBatsman.nlargest(10, "Runs")                                                                              

#print(deliveries_position)
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

mapper = deliveries.groupby(['match_id', 'inning']).batsman.apply(lambda x: dict(zip(x[~x.duplicated()], np.arange(1, len(x[~x.duplicated()])+1)))).reset_index(name = 'batting_position').rename(columns = {'level_2':'batsman'})

deliveries_position = deliveries.merge(mapper, on = ['match_id', 'inning', 'batsman'], how = 'outer')

deliveries_position["batting_position"] = np.where((deliveries_position["batting_position"] == 2), 1, deliveries_position["batting_position"])

deliveries_position = deliveries_position[(deliveries_position.batting_position == 1)]

asBatsman = deliveries_position.groupby(["batsman", "batting_position"]).apply(lambda x: pd.DataFrame(data = {"Runs": x.batsman_runs.sum(), "Matches played":x.match_id.nunique()}, index = [1])).reset_index().drop(columns = "level_2")

asBatsman[["batsman", "Runs", ]].nlargest(10, "Runs")                                                                     
import pandas as pd

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df.groupby("batsman")["batsman_runs"].agg({"runs":"sum"}).nlargest(10, "runs")
import pandas as pd

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df = df[df.inning == 2]

df.groupby(["batsman"])["batsman_runs"].agg({"runs":"sum"}).nlargest(10, "runs")
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df_deliv = pd.read_csv("../input/deliveries.csv")

df_match = pd.read_csv("../input/matches.csv")

df_match = df_match.rename(columns = {"id":"match_id"})

df_deliv = df_deliv[["match_id", "batting_team", "batsman", "batsman_runs"]]

df_deliv = pd.merge(df_deliv, df_match[["match_id", "winner"]], on = "match_id")

df_deliv = df_deliv.drop(df_deliv[df_deliv.batsman_runs == 0].index)

df_deliv = df_deliv[df_deliv["batting_team"] == df_deliv["winner"]]

df_deliv.groupby("batsman").batsman_runs.agg({"runs":"sum"}).nlargest(10, "runs")
import pandas as pd

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df = df[["batsman", "batsman_runs"]]

def func(x):

    if 6 not in x.batsman_runs.unique():

        return pd.DataFrame(data = {"Batsman": x.batsman.unique(), "Runs":x.batsman_runs.sum()}, index=[1])



df.groupby("batsman").apply(lambda x: func(x)).reset_index().drop(columns = ["batsman", "level_1"]).nlargest(10, "Runs")
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df["bowling_team"] = np.where((df["bowling_team"] == "Rising Pune Supergiants"), "Rising Pune Supergiant", df["bowling_team"])

x = df.groupby(["bowling_team", "batsman"])["batsman_runs"].agg("sum")

x.groupby(level = 0).nlargest(1).reset_index(level=0, drop = True).reset_index().sort_values("batsman_runs", ascending = False)
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



deliveries = pd.read_csv("../input/deliveries.csv")

mapper = deliveries.groupby(['match_id', 'inning']).batsman.apply(lambda x: dict(zip(x[~x.duplicated()], np.arange(1, len(x[~x.duplicated()])+1)))).reset_index(name = 'batting_position').rename(columns = {'level_2':'batsman'})

deliveries_position = deliveries.merge(mapper, on = ['match_id', 'inning', 'batsman'], how = 'outer')

deliveries_position["batting_position"] = np.where((deliveries_position["batting_position"] == 2), 1, deliveries_position["batting_position"])

deliveries_position = deliveries_position[(deliveries_position.batting_position == 1)]

deliveries_position[['batsman', 'non_striker']] = np.sort(deliveries_position[['batsman', 'non_striker']])

deliveries_position.groupby(['batsman', 'non_striker']).batsman_runs.sum().reset_index().nlargest(10, "batsman_runs")
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df1 = pd.read_csv("../input/deliveries.csv")

df1[["batsman", "non_striker"]] = np.sort(df1[["batsman", "non_striker"]], 1)

mapper = df1.groupby(["batsman", "non_striker"]).batsman_runs.agg({"runs":"sum"})

mapper.nlargest(10, "runs")
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



deliveries = pd.read_csv("../input/deliveries.csv")

deliveries["bowler_runs"] = deliveries["total_runs"] - deliveries["bye_runs"] - deliveries["legbye_runs"] - deliveries["penalty_runs"]

Runs_given = deliveries.groupby("bowler").bowler_runs.agg({"Runs":"sum", "Balls":"count"})

deliveries = deliveries[(deliveries["dismissal_kind"] != "run out") & (deliveries["dismissal_kind"] != "obstructing the field") & (deliveries["dismissal_kind"] != "retired hurt")]

bowlers = deliveries.groupby("bowler")["player_dismissed"].agg({"wickets":"count"})



bowlers = pd.merge(Runs_given, bowlers, how = "outer", left_index = True, right_index = True)

bowlers["overs"] = (np.round(bowlers["Balls"]/6, 0).astype(int)).astype(str) + "." +(bowlers["Balls"]%6).astype(str)

bowlers["strike_rate"] = np.round(bowlers.Balls/bowlers.wickets, 2)

bowlers["Average"] = np.round(bowlers.Runs/bowlers.wickets, 2)

bowlers["Economy"] = np.round(bowlers.Runs/(bowlers.Balls/6), 2)

bowlers = bowlers.drop(columns = "Balls")

#bowlers.sort_values(by = ["wickets", "Economy"], ascending = [False, True])

bowlers.nlargest(10, "wickets")
import pandas as pd

import numpy as np

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df = df[(df["is_super_over"] != 1)]

df["bowler_runs"] = df["total_runs"] - df["bye_runs"] - df["legbye_runs"] - df["penalty_runs"]

df["legitimate_balls"] = np.where(((df.noball_runs == 0)&(df.wide_runs == 0)), 1, 0)

d1 = df.groupby(["match_id", "bowler"])["bowler_runs"].agg("sum").reset_index().rename(columns = {"bowler_runs":"runs conceded"})

d  = df.groupby(["match_id", "bowler"])["legitimate_balls"].agg("sum").reset_index().rename(columns = {"legitimate_balls":"balls"})

df = df[(df["dismissal_kind"] != "run out") & (df["dismissal_kind"] != "obstructing the field") & (df["dismissal_kind"] != "retired hurt")]

d2 = df.groupby(["match_id", "bowler"])["player_dismissed"].agg("count").reset_index().reset_index().rename(columns = {"player_dismissed":"wickets"})

df3 = pd.merge(d1, d2, on = ["match_id", "bowler"], how = "outer").fillna(0).drop(columns = "index")

df3 = pd.merge(df3, d, on = ["match_id", "bowler"], how = "outer").fillna(0)#.drop(columns = "index")

df3["wickets"] = df3["wickets"].astype(int)

df3["overs"] = ((df3["balls"]/6).astype(int)).astype(str) + "." +(df3["balls"]%6).astype(str)

df3.drop(columns = ["match_id", "balls"], axis = 1, inplace = True)

df3.nlargest(10, "wickets")
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import itertools

import warnings

warnings.simplefilter(action = "ignore", category = FutureWarning)



df = pd.read_csv("../input/deliveries.csv")

df_matches = pd.read_csv("../input/matches.csv")

df = df.merge(df_matches[["id", "season", "venue"]], left_on = "match_id", right_on = "id").drop(columns = "id")

df = df[df.is_super_over == 0]

df["batting_team"] = np.where((df["batting_team"] == "Rising Pune Supergiant"), "Rising Pune Supergiants", df["batting_team"])

df["venue"] = np.where((df["venue"] == "Punjab Cricket Association IS Bindra Stadium, Mohali"), "Punjab Cricket Association Stadium, Mohali", df["venue"])

df["score"] = df.groupby(["match_id", "inning"]).total_runs.cumsum()

df["balls"] = np.where(((df.wide_runs==0)&(df.noball_runs==0)), 1, 0)

df["balls_new"] = df.groupby(["match_id", "inning"]).balls.cumsum()

df["run_rate"] = np.round((df["score"]/df["balls_new"])*6,2)

#print(df.total_runs.sum()/(2*df.match_id.nunique()))

#par_score = df.groupby(["venue", "season"]).apply(lambda x: int((x.total_runs.sum()/x.match_id.nunique())/2)).unstack().fillna(0)

#par_score

import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/matches.csv")

winners = df[df.id.isin(df.groupby("season").id.max())]

winners[["season", "winner"]]
import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

import seaborn as sns



df = pd.read_csv("../input/deliveries.csv")

df2 = pd.read_csv("../input/matches.csv")

df = df.merge(df2[["season", "id"]], how = "outer", right_on = "id", left_on = "match_id")

df = df[df.batsman_runs == 6]

sns.countplot(x = "season", data = df)

df.groupby("season").batsman_runs.count()