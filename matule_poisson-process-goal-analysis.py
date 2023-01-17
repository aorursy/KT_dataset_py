# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
path = "../input/"  #Insert path here

database = path + 'database.sqlite'

conn = sqlite3.connect(database)



#Defining the number of jobs to be run in parallel during grid search

n_jobs = 1 #Insert number of parallel jobs here



#Fetching required data tables

player_data = pd.read_sql("SELECT * FROM Player;", conn)

player_attributes_data = pd.read_sql("SELECT * FROM Player_Attributes;", conn)

team_data = pd.read_sql("SELECT * FROM Team;", conn)

team_attributes_data = pd.read_sql("SELECT * FROM Team_Attributes;", conn)

match_data = pd.read_sql("SELECT * FROM Match;", conn)

country_data = pd.read_sql("SELECT * FROM Country;", conn)

league_data = pd.read_sql("SELECT * FROM League;", conn)
(match_data['country_id'] == match_data['league_id']).all()
league_data['teams'] = len(league_data)*[[]]

for i in range(len(league_data)):

    league_data.loc[i,'teams'] = [[pd.read_sql("SELECT DISTINCT away_team_api_id FROM Match WHERE league_id = %s;" % league_data.loc[i,'id'], conn).as_matrix()[:,0].tolist()]]
def Poisson_league_analysis(matches,league,M=5.):

    league_matches = matches[matches['league_id'] == league['id']]

    teams = league['teams'][0][0]

    n_matches = len(league_matches)

    n_teams = len(teams)  

    

    home_poisson = np.sum(league_matches['home_team_goal'])/n_matches*(np.ones((n_teams,n_teams))-np.identity(n_teams))

    away_poisson = np.sum(league_matches['away_team_goal'])/n_matches*(np.ones((n_teams,n_teams))-np.identity(n_teams))

    

    for i in league_matches.index:

        match = league_matches.loc[i]

        

        home_team = teams.index(match['home_team_api_id'])

        away_team = teams.index(match['away_team_api_id'])

        

        lamH = home_poisson[home_team,away_team]

        lamA = home_poisson[home_team,away_team]

        newlamH = (M*lamA + match['home_team_goal'])/(M+1)

        newlamA = (M*lamA + match['away_team_goal'])/(M+1)

        rH = np.power(newlamH/lamH,.25)

        rA = np.power(newlamA/lamA,.25)

        

        for j in range(n_teams):

            home_poisson[home_team,j] *= rH**2

            home_poisson[j,away_team] *= rH**2

            away_poisson[away_team,j] *= rA**2

            away_poisson[j,home_team] *= rA**2

      

    return home_poisson,away_poisson,teams

        

        
results = [Poisson_league_analysis(match_data,league_data.loc[i]) for i in league_data.index]
pd.DataFrame(results[0][1])
match_data[match_data['country_id']==country_data['id'][1]][:3]
league_data
nTeams = len(team_data)

home_intensity = np.ones((nTeams,nTeams))

away_intensity = np.ones((nTeams,nTeams))