import os
os.environ["MKL_THREADING_LAYER"] = "GNU"

import pandas as pd
import numpy as np
import theano.tensor as tt
import matplotlib.pyplot as plt
import pymc3
from sklearn.preprocessing import LabelEncoder
df_regular = pd.read_csv("../input/RegularSeasonCompactResults.csv")
df_tourney = pd.read_csv("../input/NCAATourneyCompactResults.csv")
df_seeds = pd.read_csv("../input/NCAATourneySeeds.csv")
df_tourney["WLoc"].unique()
df_regular_2017 = df_regular[df_regular.Season == 2017].copy().reset_index(drop=True)
df_regular_2017.DayNum.describe()
df_regular_2017.head()
df_regular_2017["HomeScore"] = df_regular_2017.apply(
    lambda x: x["WScore"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["LScore"], axis=1)
df_regular_2017["AwayScore"] = df_regular_2017.apply(
    lambda x: x["LScore"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["WScore"], axis=1)
df_regular_2017["HomeTeam"] = df_regular_2017.apply(
    lambda x: x["WTeamID"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["LTeamID"], axis=1)
df_regular_2017["AwayTeam"] = df_regular_2017.apply(
    lambda x: x["LTeamID"] if x["WLoc"] == "H" or x["WLoc"] == "N" else x["WTeamID"], axis=1)
assert all(df_regular_2017["HomeTeam"] != df_regular_2017["AwayTeam"])
assert all(df_regular_2017["HomeScore"] != df_regular_2017["AwayScore"])
teams = sorted(list(set(df_regular_2017["HomeTeam"]) | set(df_regular_2017["AwayTeam"])))
team_encoder = LabelEncoder()
team_encoder.fit(teams)
df_regular_2017["HomeTeamID"] = team_encoder.transform(df_regular_2017["HomeTeam"])
df_regular_2017["AwayTeamID"] = team_encoder.transform(df_regular_2017["AwayTeam"])
df_regular_2017.head()
advantage = (df_regular_2017["WLoc"] != "N").astype("int")
num_teams = len(team_encoder.classes_)
num_games = df_regular_2017.shape[0]
num_teams, num_games
model = pymc3.Model()
with model:
    # global model parameters
    home = pymc3.Flat('home')
    sd_att = pymc3.HalfStudentT('sd_att', nu=3, sd=2.5)
    sd_def = pymc3.HalfStudentT('sd_def', nu=3, sd=2.5)
    intercept = pymc3.Flat('intercept')
    
    # team-specific model parameters
    offs_star = pymc3.Normal('offs_star', mu=0, sd=sd_att, shape=num_teams)
    defs_star = pymc3.Normal('defs_star', mu=0, sd=sd_def, shape=num_teams)
    offs = pymc3.Deterministic('offs', offs_star - tt.mean(offs_star))
    defs = pymc3.Deterministic('defs', defs_star - tt.mean(defs_star))
    
    # derive the scoring intensity for a game
    home_theta = tt.exp(
        intercept + home * advantage + offs[df_regular_2017["HomeTeamID"].values] + defs[df_regular_2017["AwayTeamID"].values])
    away_theta = tt.exp(
        intercept  + offs[df_regular_2017["AwayTeamID"].values] + defs[df_regular_2017["HomeTeamID"].values])
    
    # likelihood of observed data
    home_points = pymc3.Poisson('home_points', mu=home_theta, observed=df_regular_2017["HomeScore"].values)
    away_points = pymc3.Poisson('away_points', mu=away_theta, observed=df_regular_2017["AwayScore"].values)
with model:
    trace = pymc3.sample(2000, tune=1000)
pymc3.traceplot(trace)
plt.show()
pymc3.forestplot(trace, varnames=["offs"], main="Team Offense")
plt.show()
pymc3.forestplot(trace, varnames=["defs"], main="Team defense")
plt.show()
team_mean_offs = np.mean(trace['offs'], axis=0)
team_mean_defs = np.mean(trace['defs'], axis=0)
def calculate_winning_probability(trace, team_1=0, team_2=1, sample_size=100):
    draw = np.random.randint(0, trace['intercept'].shape[0], size=sample_size)
    intercept_ = trace['intercept'][draw]
    offs_ = trace['offs'][draw]
    defs_ = trace['defs'][draw]
    home_theta_ = np.exp(intercept_ + offs_[:, team_1] + defs_[:, team_2])
    away_theta_ = np.exp(intercept_ + offs_[:, team_2] + defs_[:, team_1])
    home_score_ = np.random.poisson(home_theta_, sample_size)
    away_score_ = np.random.poisson(away_theta_, sample_size)   
    wins = np.mean((home_score_ - away_score_ > 0))
    return wins, (team_mean_offs[team_1], team_mean_defs[team_1]), (team_mean_offs[team_2], team_mean_defs[team_2])
calculate_winning_probability(trace, 0, 1, 5000)
calculate_winning_probability(trace, 2, 10, 1000)
df_tourney_2017 = df_tourney[df_tourney.Season == 2017].copy().reset_index(drop=True)
df_tourney_2017["WTeamID"] = team_encoder.transform(df_tourney_2017.WTeamID)
df_tourney_2017["LTeamID"] = team_encoder.transform(df_tourney_2017.LTeamID)
df_tourney_2017.head()
calculate_winning_probability(trace, 135, 334, 1000)
calculate_winning_probability(trace, 182, 199, 1000)
%time df_tourney_2017["Pred"] = df_tourney_2017.apply(lambda x: calculate_winning_probability(trace, x["WTeamID"], x["LTeamID"], 1000)[0], axis=1)
_ = plt.hist(df_tourney_2017["Pred"], bins=20)
plt.show()
sum(df_tourney_2017["Pred"] > 0.5) / df_tourney_2017.shape[0]
np.mean(-np.log(df_tourney_2017["Pred"].values))

