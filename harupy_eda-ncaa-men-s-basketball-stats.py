import os

import traceback

from glob import glob



import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import pandas as pd

import seaborn as sns

sns.set()

import plotly.offline as py

py.init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.tools as tls

import plotly.figure_factory as ff

from plotly.colors import DEFAULT_PLOTLY_COLORS



import ipywidgets as widgets

from ipywidgets import interact, interact_manual



from IPython.display import display



SMALL_SIZE = 14

MEDIUM_SIZE = 16

BIGGER_SIZE = 24



plt.rc('font', size=SMALL_SIZE)

plt.rc('axes', titlesize=BIGGER_SIZE)

plt.rc('axes', labelsize=MEDIUM_SIZE)

plt.rc('xtick', labelsize=SMALL_SIZE)

plt.rc('ytick', labelsize=SMALL_SIZE)

plt.rc('legend', fontsize=SMALL_SIZE)

plt.rc('figure', titlesize=BIGGER_SIZE)
%ls ../input
def lower_first_char(s):

    return s[:1].lower() + s[1:] if s else ''



def is_number(s):

    try:

        float(s)

        return True

    except ValueError:

        return False



def inspect_features(df):

    df_insp = pd.DataFrame({}, index=df.columns)

    df_insp["dtype"] = df.dtypes

    df_insp["null count"] = df.isnull().sum()

    df_insp["isspace count"] = [df[col].str.isspace().sum().astype(int)

                                if df[col].dtype == "object" else 0 for col in df.columns]

    df_insp["numeric count"] = [df[col].apply(is_number).sum() for col in df.columns]

    df_insp["unique count"] = df.nunique()

    df_insp["unique values (only showing top 5)"] = [df[col].unique()[:5] for col in df.columns]

    return df_insp
data_file_paths = glob("../input/datafiles/*.csv")



@interact

def show_df(column=data_file_paths):

    print(column)

    df = pd.read_csv(column, encoding="ISO-8859-1")

    display(inspect_features(df))
df_teams = pd.read_csv("../input/datafiles/Teams.csv")

df_teams.index = df_teams["TeamID"]

df_teams.head()
df_coaches = pd.read_csv("../input/datafiles/TeamCoaches.csv")

ck_first_year = df_coaches[df_coaches["CoachName"].str.contains("mike_krzyzewski")]["Season"].min()

print("Coach K's first year (?):", ck_first_year)
df_cr = pd.read_csv("../input/datafiles/RegularSeasonCompactResults.csv")

df_cr["PointDifference"] = df_cr["WScore"] - df_cr["LScore"]

idx_max_pd = df_cr["PointDifference"].idxmax()

print("Biggest point difference:", df_cr.loc[idx_max_pd, "PointDifference"])

df_cr.loc[[idx_max_pd], :]
df_cr.groupby("Season")[["WScore", "LScore"]].mean().plot(marker="o", title="Scoring Trend", figsize=(20, 6)).set_ylabel("Average Score");
pbp_paths = glob("../input/playbyplay_201*/*.csv")

@interact

def show_df(column=pbp_paths):

    print(column)

    df = pd.read_csv(column, encoding = "ISO-8859-1")

    # event log file is large so it might take time

    display(inspect_features(df))

    display(df.head())
import re



def get_player_stats(player_name, year):

    """

    Event file is large so it might take time to compute the results

    """

    df_player = pd.read_csv(f"../input/playbyplay_{year}/Players_{year}.csv")

    if not df_player["PlayerName"].str.contains(player_name).any():

        print(f"{player_name} is not found in Players_{year}.csv")

        return

    df_player.index = df_player["PlayerID"]

    df_event = pd.read_csv(f"../input/playbyplay_{year}/Events_{year}.csv")

    

    df_event["PlayerName"] =  df_player.loc[df_event["EventPlayerID"]]["PlayerName"].values

    df_event["WTeam"] = df_teams.loc[df_event["WTeamID"]]["TeamName"].values

    df_event["LTeam"] = df_teams.loc[df_event["LTeamID"]]["TeamName"].values

    df_event["EventTeam"] = df_teams.loc[df_event["EventTeamID"]]["TeamName"].values

    df_event["OpponentTeam"] = ""

    did_event_team_win = df_event["EventTeamID"] == df_event["WTeamID"]

    df_event["OpponentTeam"][did_event_team_win] = df_event["LTeam"][did_event_team_win]

    df_event["OpponentTeam"][~did_event_team_win] = df_event["WTeam"][~did_event_team_win]

    df_event["Win"] = did_event_team_win.astype(int)

    

    df_stats = df_event[df_event["PlayerName"].str.contains(player_name)]

    event_team_id = df_stats["EventTeamID"].values[0]

    lteam_id = df_stats["LTeamID"].values[0]

    day_played = df_stats["DayNum"].unique()

    game_points = (

        df_event[["DayNum", "WPoints", "LPoints"]]

        [

            df_event["DayNum"].isin(day_played) &

            (df_event["WPoints"] > 0) &

            ((df_event["WTeamID"] == event_team_id) | (df_event["LTeamID"] == event_team_id))

        ]

        .drop_duplicates("DayNum", keep="last").iloc[:, 1:].values

    )

    df_stats = df_stats.groupby(["PlayerName", "EventTeam", "DayNum", "OpponentTeam", "Win", "EventType"]).size().unstack().fillna(0).astype(int)

    rebounds = df_stats["reb_def"] + df_stats["reb_off"]

    

    point_cols = [col for col in df_stats.columns if "made" in col]

    points = 0



    for point_col in point_cols:

        num = int(re.search(r"\d", point_col).group(0))

        points += num * df_stats[point_col].values



    df_stats.insert(0, "turnover", df_stats.pop("turnover"))

    df_stats.insert(0, "block", df_stats.pop("block"))

    df_stats.insert(0, "assist", df_stats.pop("assist"))

    df_stats.insert(0, "rebounds", rebounds)

    df_stats.insert(0, "points", points)

    df_stats.insert(0, "lpoints", game_points[:, 1])

    df_stats.insert(0, "wpoints", game_points[:, 0])

    

    return df_stats
get_player_stats("FULTZ_MARKELLE", 2017)
get_player_stats("TATUM_JAYSON", 2017)
get_player_stats("BALL_LONZO", 2017)
get_player_stats("FOX_DEAARON", 2017)
get_player_stats("PARKER_JABARI", 2014)