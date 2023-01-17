import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

from sklearn import linear_model

import matplotlib.pyplot as plt

%matplotlib inline
nba = pd.read_csv('../input/NBA Game Data 2016-2017.csv')
weekday = nba.groupby(by=['Week Day', 'Home/Away'])

weekday_home_away = weekday['Tm'].mean()

weekday_home_away
weekday_home_away = weekday_home_away.unstack()

weekday_home_away
weekday_home_away["Home Team Margin"] = weekday_home_away['Home']-weekday_home_away['Away']

weekday_home_away
weekday_home_away.reset_index(inplace = True)

weekday_home_away
weekday_home_away["DayID"] = [5,1,6,0,4,2,3]
weekday_home_away.sort_values("DayID")
weekday_home_away.set_index("DayID", inplace = True)

weekday_home_away.sort_index(inplace = True)
weekday_home_away.plot(x='Week Day', y='Home Team Margin', kind="Bar", color="Red");
nba["Net Points"] = nba["Tm"] - nba["Opp"]

nba.head()
team_summary = nba.groupby("Team")

net_points = team_summary["Net Points"].sum()

wins = team_summary["Result"].value_counts().unstack()
wins_net_points = wins.join(net_points)

wins_net_points
wins_net_points[["W", "Net Points"]].corr()["W"]["Net Points"]
lm = linear_model.LinearRegression()

x = wins_net_points["Net Points"].values.reshape(-1, 1)

y = wins_net_points["W"].values.reshape(-1, 1)

model = lm.fit(x,y)
lm.score(x,y)

# R^2
intercept = lm.intercept_

coef = lm.coef_
team_week_day = nba.groupby(["Team","Home/Away","Week Day"]).count()["G"]
weekday_home_away.set_index('Week Day', inplace = True)
team_home_away = pd.DataFrame(team_week_day).join(pd.DataFrame(weekday_home_away['Home Team Margin']))
team_home_away.reset_index(inplace = True)
team_home_away.loc[team_home_away["Home/Away"] == "Away",["Home Team Margin"]] = team_home_away.loc[team_home_away["Home/Away"] == "Away", ["Home Team Margin"]] * -1
team_home_away["Impact"]=team_home_away['G']*team_home_away["Home Team Margin"]
team_impact=team_home_away.groupby("Team")
point_impact = team_impact["Impact"].sum()
game_impact = point_impact * coef[0][0]
game_impact.sort_values().plot(kind="Bar", color="Blue", figsize=(10,5));