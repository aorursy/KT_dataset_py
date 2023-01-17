%matplotlib inline

import numpy as np
import matplotlib as mplc
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine # database connection
import numpy as np

from bokeh.models.widgets import Panel, Tabs, Dropdown
from bokeh.models import  Callback, CustomJS

from bokeh.plotting import *
from bokeh import mpl

sns.set_style("dark",{'axes.facecolor':'black', 'figure.facecolor':'black', 'grid.color':'darkgrey'})
output_notebook()
engine  = create_engine("sqlite:///../input/database.sqlite")
leagues = pd.read_sql_query('SELECT * FROM League;', engine)
matches = pd.read_sql_query('SELECT * FROM Match;', engine)
sub_cols = matches.columns[:55]
matches = matches[sub_cols].dropna()
teams = pd.read_sql_query('SELECT * FROM Team;', engine)
matches.groupby(['home_player_X1', 'home_player_Y1']).size()
# If the dataset is too large to create the visualisation for all teams, 
# use this function to subset to a single league
def select_league_matches(matches, leagues, league, season=None):
    league_id = int(leagues[leagues['name'] == league]['id'])
    s_cond = True if season == None else matches['season'] == season
    return matches[(matches['league_id'] == league_id) & (s_cond)]

# Extract the (X, Y) position of the i-th player based on prefix
# which represents either 'Home' or 'Away'
def extract_pos(match, prefix, i):
    X = float(match[prefix + 'X' + str(i)])
    Y = float(match[prefix + 'Y' + str(i)])
    return [X, Y]

# Extract the formations of the home and away teams as well as their ids from
# a single match
def extract_formation(match):
    home_id = int(match['home_team_api_id'])
    away_id = int(match['away_team_api_id'])
    home_form = []
    away_form = []
    h_range = (10, 0)
    a_range = (10, 0)
    for i in range(1, 11):
        h_pos = extract_pos(match, 'home_player_', i)
        a_pos = extract_pos(match, 'away_player_', i)
        if i != 1:
            h_range = min(h_range[0], h_pos[0]), max(h_range[1], h_pos[1])
            a_range = min(a_range[0], a_pos[0]), max(a_range[1], a_pos[1])
        home_form.append(h_pos)
        away_form.append(a_pos)
    home_form[0][0] = 5#np.mean(h_range)
    away_form[0][0] = 5#np.mean(a_range)
    return (home_form, away_form, home_id, away_id)

# Extract the formations of all teams in the matches table passed as an argument
# and return a dictionary consisting of all formations used by each team in the
# matches dataframe
def extract_formations(matches):
    formations = {}
    for i in matches.index:
        h_form, a_form, h_id, a_id = extract_formation(matches.loc[i])
        if h_id in formations.keys():
            formations[h_id] += h_form
        else:
            formations[h_id] = h_form
        if a_id in formations.keys():
            formations[a_id] += a_form
        else:
            formations[a_id] = a_form
    return formations

# Extract the formations of all teams in the matches table, and substitute the
# team-ids with the team name as the key to the dictionary
def extract_formations_team_names(matches, teams):
    formations = extract_formations(matches)
    keys = list(formations.keys())
    for i in keys:
        name = teams.loc[teams['team_api_id'] == i,'team_long_name'].values[0]
        formations[name] = formations.pop(i)
    return formations
sub_matches = select_league_matches(matches, leagues, 'England Premier League')
formations = extract_formations_team_names(sub_matches, teams)
def generate_plot(formation, team_name):
    mplc.rc("figure", figsize=(8, 8))
    data = pd.DataFrame(formation, columns=["X", "Y"])
    sns.kdeplot(data, shade=False, n_levels=15, cmap="RdBu_r")
    plt.title(team_name + " Formation")
    return mpl.to_bokeh()

d_keys = list(formations.keys())
show(generate_plot(formations[d_keys[0]], d_keys[0]))
def visualise(formations):
    panels = []
    team_names = sorted(formations.keys())
    for i in team_names:
        p = generate_plot(formations[i], i)
        panels.append(Panel(child=p, title=i))

    tabs = Tabs(tabs=panels)

    callback = CustomJS(args={'tabs':tabs}, code="""
        debugger;
        menu = cb_obj.get('menu');
        value = cb_obj.get('value');
        val_loc = 0;
        for (i = 0; i < menu.length; ++i)
            if (menu[i][0] === value)
                break;
        val_loc = i;
        tabs.set('active', val_loc);
        cb_obj.set('label', 'Team:' + value);
    """)

    dropdown = Dropdown(label="Team:", type='warning', menu=list(zip(team_names, team_names)), callback=callback)
    return vplot(dropdown, tabs)
from IPython.display import HTML

hide_tabs = """
    <style>.bk-bs-nav-tabs { height: 0; width:0; opacity=0;text-indent: -9999px}</style>
    <style>.bk-bs-nav-tabs>li { height: 0; width:0; opacity=0;text-indent: -9999px}</style>
    <style>.bk-bs-nav-tabs>li>a { height: 0; width:0; opacity=0;text-indent: -9999px}</style>
"""
HTML(hide_tabs)
show(visualise(formations))
