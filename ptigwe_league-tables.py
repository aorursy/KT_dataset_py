import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine # database connection
import numpy as np
from IPython.display import display, clear_output
from itertools import product

from bokeh.models import  Callback, ColumnDataSource, Rect, Select,CustomJS
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn, Dropdown

from bokeh.plotting import *
output_notebook()
engine  = create_engine("sqlite:///../input/database.sqlite")
countries = pd.read_sql_query('SELECT * FROM Country;', engine)
countries.rename(columns={'id':'country_id', 'name':'Country'}, inplace=True)
leagues = pd.read_sql_query('SELECT * FROM League;', engine)
leagues.rename(columns={'id':'league_id', 'name':'League'}, inplace=True)
matches = pd.read_sql_query('SELECT * FROM Match;', engine)
matches = matches[matches.columns[:11]]
teams = pd.read_sql_query('SELECT * FROM Team;', engine)
master = pd.merge(matches, leagues, on=['league_id', 'country_id'], how='left')
master = pd.merge(master, countries, on='country_id', how='left')
master.head()
seasons = master.season.unique().tolist()
league_names = leagues['League'].tolist()
def construct_table(master, teams, league, season):
    cur_master = master[(master['League'] == league) & (master['season'] == season)]
    cur_teams = cur_master['home_team_api_id'].unique()
    #cur_teams = teams[teams['team_api_id'].isin(cur_teams)]
    cols = ['Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA']
    table = teams[teams['team_api_id'].isin(cur_teams)].copy()
    for i in cols:
        table[i] = 0
    for i in cur_master.index:
        cur = cur_master.loc[i]
        home_team = table['team_api_id'] == cur['home_team_api_id']
        away_team = table['team_api_id'] == cur['away_team_api_id']
        
        table.loc[home_team, 'Played'] += 1
        table.loc[away_team, 'Played'] += 1
        
        home_goals = cur['home_team_goal']
        away_goals = cur['away_team_goal']
        
        table.loc[home_team, 'GF'] += home_goals
        table.loc[home_team, 'GA'] += away_goals
        table.loc[away_team, 'GF'] += away_goals
        table.loc[away_team, 'GA'] += home_goals
        
        if home_goals == away_goals:
            table.loc[home_team, 'Drawn'] += 1
            table.loc[away_team, 'Drawn'] += 1
        elif home_goals > away_goals:
            table.loc[home_team, 'Won'] += 1
            table.loc[away_team, 'Lost'] += 1
        else:
            table.loc[home_team, 'Lost'] += 1
            table.loc[away_team, 'Won'] += 1
        
        table['GD'] = table['GF'] - table['GA']
        table['Points'] = 3 * table['Won'] + table['Drawn']
        table.sort_values(['Points', 'GD', 'GF', 'GA'], ascending=False, inplace=True)
        table.index = range(1,len(table) + 1)
        
    return table
print(construct_table(master, teams, 'England Premier League', seasons[-1]))
league_seasons = list(product(league_names, seasons))
cols = ['team_long_name', 'Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points']
table = construct_table(master, teams, 'Belgium Jupiler League', seasons[0])[cols]
for i in league_seasons:
    tmp = construct_table(master, teams, i[0], i[1])[cols]
    new_cols = [i[0] + i[1] + j for j in tmp.columns]
    tmp.columns = new_cols
    table = pd.concat([table, tmp], axis=1)
src1 = ColumnDataSource(table)
src2 = ColumnDataSource(table[cols].dropna())
columns = [TableColumn(field=i, title=i) for i in cols]
dt = DataTable(source=src2, columns=columns)

callback = CustomJS(args={'src1':src1, 'src2':src2, 'dt':dt}, code="""
    cols = ['team_long_name', 'Played', 'Won', 'Drawn', 'Lost', 'GF', 'GA', 'GD', 'Points'];
    ls_val = cb_obj.get('value');
    var arrayLength = cols.length;
    data1 = src1.get('data');
    data2 = src2.get('data');
    
    for (i = 0; i < arrayLength; i++) {
        data2[cols[i]] = data1[ls_val + cols[i]];
    }
    
    cb_obj.attributes.label = 'League & Season:' + ls_val;
    src1.trigger('change');
    src2.trigger('change');
    cb_obj.trigger('change');
    dt.trigger('change');
""")

ls_str = [i[0] + i[1] for i in league_seasons]
dropdown = Dropdown(label="League & Season", type='warning', menu=list(zip(ls_str, ls_str)), callback=callback)
show(vplot(dropdown, dt))
