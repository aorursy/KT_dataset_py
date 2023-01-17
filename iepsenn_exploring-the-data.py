import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from typing import List

import matplotlib.pyplot as plt



import os

print(os.listdir("../input/dataset/"))

print(os.listdir("../input/dataset/2012"))

def load_df(year):

    path = "../input/dataset/" + str(year) + "/"

    games_path = path + "games.csv"

    table_path = path + "table.csv"

    games = pd.read_csv(games_path, index_col=0)

    table = pd.read_csv(table_path, index_col=0)

    

    return games, table
def transform_table(tables:List[pd.DataFrame]):

    for t in tables:

        t.rename(columns={'Classificação':'Team'}, inplace=True)

        team_names = []

        for team in t.Team.values:

            team_names.append(team[-3:])

        t.Team = team_names

    return tables
def load_data():

    tables = []



    for year in range(7):

        _, t = load_df(year+2012)

        tables.append(t)

    

    tables = transform_table(tables)

    

    return tables
tables = load_data()
team_points = {}

for table in tables:

    for team, point in zip(table.Team, table.P):

        if team in team_points:

            team_points[team] += point

        else:

            team_points[team] = point

points_by_team = pd.DataFrame.from_dict(data=team_points, orient='index', columns=["Points"])

#points_by_team['Team'] = points_by_team.index

#index = np.arange(points_by_team.shape[0])

#points_by_team.index = index

#points_by_team = points_by_team[['Team', 'Points']]

points_by_team.sort_values(by="Points", ascending=False)[:10].plot.bar()

plt.title("teams that scored more points")

plt.show()
points_by_team.sort_values(by="Points", ascending=True)[:10].plot.bar()

plt.title("teams that scored fewer points")

plt.show()
team_wins = {}

for table in tables:

    for team, wins in zip(table.Team, table.V):

        if team in team_wins:

            team_wins[team] += wins

        else:

            team_wins[team] = wins

wins_by_team = pd.DataFrame.from_dict(data=team_wins, orient='index', columns=["Wins"])

wins_by_team.sort_values(by="Wins", ascending=False)[:10].plot.bar()

plt.title("teams that won more matches")

plt.show()
team_losses = {}

for table in tables:

    for team, losses in zip(table.Team, table.D):

        if team in team_losses:

            team_losses[team] += losses

        else:

            team_losses[team] = losses

losses_by_team = pd.DataFrame.from_dict(data=team_losses, orient='index', columns=["Losses"])

losses_by_team.sort_values(by="Losses", ascending=False)[:10].plot.bar()

plt.title("teams that lost more matches")

plt.show()
team_draws = {}

for table in tables:

    for team, draws in zip(table.Team, table.E):

        if team in team_draws:

            team_draws[team] += draws

        else:

            team_draws[team] = draws

draws_by_team = pd.DataFrame.from_dict(data=team_draws, orient='index', columns=["Draws"])

draws_by_team.sort_values(by="Draws", ascending=False)[:10].plot.bar()

plt.title("teams with more draws")

plt.show()
team_goalsfor = {}

for table in tables:

    for team, goals in zip(table.Team, table.G):

        if team in team_goalsfor:

            team_goalsfor[team] += goals

        else:

            team_goalsfor[team] = goals

goalsfor_by_team = pd.DataFrame.from_dict(data=team_goalsfor, orient='index', columns=["Goals"])

goalsfor_by_team.sort_values(by="Goals", ascending=False)[:10].plot.bar()

plt.title("teams that scored more goals")

plt.show()
goalsfor_by_team.sort_values(by="Goals", ascending=True)[:10].plot.bar()

plt.title("teams that scored fewer goals")

plt.show()
team_goalsagainst = {}

for table in tables:

    for team, goals in zip(table.Team, table.GC):

        if team in team_goalsagainst:

            team_goalsagainst[team] += goals

        else:

            team_goalsagainst[team] = goals

goalsagainst_by_team = pd.DataFrame.from_dict(data=team_goalsagainst, orient='index', columns=["Goals"])

goalsagainst_by_team.sort_values(by="Goals", ascending=False)[:10].plot.bar()
teams = points_by_team.sort_values(by="Points", ascending=False)[:3].index

points = []

wins = []

draws = []

losses = []

goals_for = []

goals_against = []



for team in teams:

    points.append(team_points[team])

    wins.append(team_wins[team])

    draws.append(team_draws[team])

    losses.append(team_losses[team])

    goals_for.append(team_goalsfor[team])

    goals_against.append(team_goalsagainst[team])



d = {"Team" : teams, "Points" : points, "Wins" : wins, "Draws" : draws, "Losses" : losses, "Goals_For" : goals_for, "Goals_against": goals_against}   

teams_stats = pd.DataFrame(data=d)

teams_stats
from plotly import tools

import plotly.plotly as py

from plotly.offline import init_notebook_mode, iplot

init_notebook_mode(connected=True)

import plotly.graph_objs as go

import plotly.figure_factory as ff

from IPython.display import HTML, Image
labels = teams_stats.columns[1:]



team1 = go.Scatterpolar(

    r = teams_stats.sort_values(by='Points', ascending=False).iloc[0,:],

    theta = labels,

    fill = "toself",

    name = teams[0]

)



team2 = go.Scatterpolar(

    r = teams_stats.sort_values(by='Points', ascending=False).iloc[1,:],

    theta = labels,

    fill = "toself",

    name = teams[1]

)



team3 = go.Scatterpolar(

    r = teams_stats.sort_values(by='Points', ascending=False).iloc[2,:],

    theta = labels,

    fill = "toself",

    name = teams[2]

)



data = [team1, team2, team3]



layout = go.Layout(

    polar = dict(

        radialaxis = dict(

            visible = True,

            range = [0, 600]

        )

    ),

    showlegend = True,

    title = "Teams that scored more points"

)



fig = go.Figure(data=data, layout=layout)

iplot(fig, filename="Stats")