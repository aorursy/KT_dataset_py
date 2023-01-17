import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

pd.set_option('display.max_columns', None)
# set variables
TEAM_NAME = "Chelsea"
SEASON = "2008/2009"
# create engine to the SQLite database
engine = create_engine("sqlite:////kaggle/input/soccer/database.sqlite")
# fetch team info
team = pd.read_sql_query("SELECT * FROM Team WHERE team_long_name LIKE '%" + TEAM_NAME + "%';", engine).loc[0]
team_id = int(team["team_api_id"])
print(team)
# fetch country info based on selected team
country = pd.read_sql_query("SELECT * FROM Match WHERE home_team_api_id = " + str(team_id) + ";", engine).loc[0]
country
# get country id
country_id = int(country["country_id"])
# get all matches of season for country
matches = pd.read_sql_query(
    "SELECT * FROM Match WHERE country_id = " + str(country_id) + " AND season = '" + SEASON + "';", engine)
matches.head()
# filter team's matches
matches_home = matches[matches["home_team_api_id"] == team_id]
matches_away = matches[matches["away_team_api_id"] == team_id]
def get_player_position(match, team_situation, idx):
    if team_situation == "home":
        prefix = "home_player_"
    else:
        prefix = "away_player_"
    X = float(match[prefix + 'X' + str(idx)])
    Y = float(match[prefix + 'Y' + str(idx)])
    return (X, Y)
def get_match_formation(match, team_situation):
    formation = []
    # exclude the goalkeeper
    for i in range(2, 12):
        formation.append(get_player_position(match, team_situation, i))
    return formation
def get_all_season_formations(matches, team_situation):
    formations = []
    for idx in matches.index:
        match = matches.loc[idx]
        formations.extend(get_match_formation(match, team_situation))
    return formations
# fetch formations for both home and away matches
home_formations = get_all_season_formations(matches_home, "home")
away_formations = get_all_season_formations(matches_away, "away")
def show_formations(team_full_name, home, away):
    # show plots side by side
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].set_title(team_full_name + " Home Formation", fontdict={'fontsize': 20, 'fontweight': 500})
    axes[0].axis("off")
    data = pd.DataFrame(home, columns=['X', 'Y'])
    sns.kdeplot(data['X'], data['Y'], shade=True, n_levels=12, cmap="RdBu_r", ax=axes[0])
    axes[1].set_title(team_full_name + " Away Formation", fontdict={'fontsize': 20, 'fontweight': 500})
    axes[1].axis("off")
    data = pd.DataFrame(away, columns=['X', 'Y'])
    sns.kdeplot(data['X'], data['Y'], shade=True, n_levels=12, cmap="RdBu_r", ax=axes[1])
show_formations(team["team_long_name"], home_formations, away_formations)
