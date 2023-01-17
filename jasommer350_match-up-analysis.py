import numpy as np

import pandas as pd

#import xml.etree.ElementTree as ET

import sqlite3  

from bokeh.charts import BoxPlot, Bar, output_notebook, show

from bokeh.models import HoverTool

from bokeh.plotting import figure, ColumnDataSource               
output_notebook()
query = """SELECT * FROM Match a

           INNER JOIN (SELECT team_long_name as home_name, team_api_id AS t1_id FROM Team) b ON a.home_team_api_id = b.t1_id

           INNER JOIN (SELECT team_long_name as away_name, team_api_id AS t2_id FROM Team) c ON a.away_team_api_id = c.t2_id

           WHERE a.league_id=1729;"""



#load data (make sure you have downloaded database.sqlite)

with sqlite3.connect('../input/database.sqlite') as con:

    matches_all = pd.read_sql_query(query, con)

    

#Selecting Columns

matches = matches_all[['match_api_id','home_team_api_id','home_name','away_team_api_id','away_name', 'season', 'stage','home_team_goal','away_team_goal']]



#Setup list for collecting the matchup combinations

matchups = list()
def matchStats(row):

    home_team, home_name, away_team, away_name, home_goals, away_goals = (row.home_team_api_id, row.home_name, row.away_team_api_id, row.away_name, row.home_team_goal,row.away_team_goal)

    #Setting Up different scenarios

    winLossScenarios = {'t1_hwin': (1., 0., 0., 0., 0., 0., 0. ,1.,0. ),

                       't1_hloss': (0., 1., 0., 0., 0., 0., 1. ,0.,0. ),

                       't2_hwin':  (0., 0., 0., 1., 1., 0., 0. ,0.,0. ),

                       't2_hloss': (0. , 0., 1., 0., 0., 1., 0. ,0.,0.),

                       'draw': (0., 0., 0., 0., 0., 0., 0. ,0.,1.)}

    #Set default as home win at first

    result = winLossScenarios['t1_hwin']

    #Setting default t1 and t2

    t1 = home_team

    t1_name = home_name

    t2 = away_team

    t2_name = away_name

    t1_goals = home_goals

    t2_goals = away_goals

    homeField = home_team

    if (away_team, home_team) in matchups:

        #checking if the team combination has been done before

        t1 = away_team

        t1_name = away_name

        t2 = home_team

        t2_name = home_name

        t2_goals = home_goals

        t1_goals = away_goals

        homeField = t2 #update the home field team

    elif (home_team, away_team) not in matchups:

        #Cannot find matchup must be new so set to home team, away team

        matchups.append((home_team, away_team))

    #Check which team won    

    t1_goal_diff = t1_goals - t2_goals

    if t1_goal_diff < 0:

        #T1 Losses, was T1 home or away?

        if t1 == homeField:

            result = winLossScenarios['t1_hloss']

        else:

            result = winLossScenarios['t2_hwin']

    elif t1_goal_diff == 0:

        result = winLossScenarios['draw']

    else:

        #T1 Wins, and they are not home team

        if t2 == homeField:

            result = winLossScenarios['t2_hloss']

    t1_goals = t1_goals*1.0

    t2_goals = t2_goals*1.0

    return (t1, t1_name, t1_goals, t2, t2_name, t2_goals) + result
matches.loc[:, 't1'], matches.loc[:, 't1_name'], matches.loc[:, 't1_goals'], matches.loc[:, 't2'], matches.loc[:, 't2_name'], matches.loc[:, 't2_goals'], matches.loc[:, 't1_hwin'], matches.loc[:, 't1_hloss'], matches.loc[:, 't1_awin'], matches.loc[:, 't1_aloss'], matches.loc[:, 't2_hwin'], matches.loc[:, 't2_hloss'], matches.loc[:, 't2_awin'], matches.loc[:, 't2_aloss'], matches.loc[:, 'm_draw'] = zip(*matches.apply (lambda row: matchStats (row),axis=1))
matches.loc[:, 't1wins'] = matches.loc[:, 't1_hwin'] + matches.loc[:, 't1_awin']

matches.loc[:, 't2wins'] = matches.loc[:, 't2_hwin'] + matches.loc[:, 't2_awin']

matches.loc[:, 'homeWins'] = matches.loc[:, 't1_hwin'] + matches.loc[:, 't2_hwin']

matches.loc[:, 'awayWins'] = matches.loc[:, 't1_awin'] + matches.loc[:, 't2_awin']



matches.loc[:, 'winDiff'] = matches.loc[:, 't1wins'] - matches.loc[:, 't2wins'] 

matches.loc[:, 'goalDiff'] = matches.loc[:, 't1_goals'] - matches.loc[:, 't2_goals']

matches.loc[:, 'totalMatchGoals'] = matches.loc[:, 't1_goals'] + matches.loc[:, 't2_goals']

#matches['winDiff'] = matches['winDiff'].astype(float) 

matches.loc[:, 'matchTeams'] = matches.loc[:, 't1_name'] + ' - ' + matches.loc[:, 't2_name'] 
matchGoal_df = matches.groupby(['season'])[['totalMatchGoals']].mean()

matchGoal_df
pBox = BoxPlot(matches, values='totalMatchGoals', label='season',

            title="Total Match Goals Summary (grouped by Season)")



show(pBox)
aggregations = {

    'totalMatchGoals': {

        'games_played': 'count',

        'avg_goals': 'mean',

        'total_goals': 'sum'

    },

    'goalDiff': {     

        'goal_diff': 'sum'

    },

    't1_goals': {

        't1_goals': 'sum'

    },

    't2_goals': {

        't2_goals': 'sum'

    }

}

 

# Perform groupby aggregation by "month", but only on the rows that are of type "call"

matchGoal_summary = matches.groupby('matchTeams').agg(aggregations).reset_index()

matchGoal_summary.columns = matchGoal_summary.columns.droplevel()

matchGoal_summary.rename(columns = {'':'MatchUp'}, inplace = True)
source = ColumnDataSource(

        data=dict(

            x=matchGoal_summary.loc[:,'total_goals'],

            y=matchGoal_summary.loc[:,'goal_diff'],

            matchup=matchGoal_summary.loc[:,'MatchUp'],

            games_played=matchGoal_summary.loc[:,'games_played'],

        )

    )



hover = HoverTool(

        tooltips=[

            ("index", "$index"),

            ("TG,GF", "@x, @y"),

            ("Games Played", "@games_played"),

            ("Match Up", "@matchup")

        ]

    )





scatter = figure(plot_width=800, plot_height=600, tools=[hover],

                 x_axis_label="Total Goals",

                 y_axis_label="Goal Difference",

                 title="Avg Goals Per Game vs Games Played")



scatter.circle('x', 'y', size=5, source=source)



#scatter = Scatter(source, x='x', y='y',

#                  title="Avg Goals Per Game vs Games Played", xlabel="Total Goals",

#                  ylabel="Goal Difference", tools=[hover])

show(scatter)
#Home Wins vs Away Wins by Season

matchHAWins_df = matches.groupby(['season'])[['homeWins', 'awayWins']].sum()

matchHAWins_df
#Home Wins vs Away Wins by Season

matchHAWins_df = matches.groupby(['matchTeams'])[['homeWins', 'awayWins', 'winDiff', 't1_hwin', 't2_hwin']].sum()

matchHAWins_df.sort('homeWins', ascending=False).head(10)
matchHAWins_df.sort('awayWins', ascending=False).head(10)
winDiff_df = matches.groupby(['matchTeams','season'], sort=False)[['winDiff', 't1wins', 't2wins']].sum().reset_index()

winDiff_df_sorted = winDiff_df.sort_values('winDiff', ascending=False) 

winDiff_top = winDiff_df_sorted[(winDiff_df_sorted['winDiff']>=1) | (winDiff_df_sorted['winDiff']<=-1)]
p = Bar(winDiff_top, label='matchTeams', values='winDiff', stack='season',

        title="Top Win Differences by Season")



show(p)