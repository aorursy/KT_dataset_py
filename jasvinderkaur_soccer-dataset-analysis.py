# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# Import useful libraries

import numpy as np 
import pandas as pd 
import sqlite3
import seaborn as sns
import matplotlib.pyplot as plt

"""import matplotlib and seaborn for visualization"""
sns.set_style("whitegrid") 
%matplotlib inline 
path = '/kaggle/input/soccer/database.sqlite'
conection = sqlite3.connect(path)
tables = pd.read_sql("""SELECT *
                        FROM sqlite_master
                        WHERE type='table';""", conection) # executing SQL query in Pandas
tables # there are 7 tables in the database 
league = pd.read_sql("""SELECT country.name AS Country,
                               league.name AS League
                        FROM league 
                        INNER JOIN country ON league.country_id=country.id;""", conection)
league # there are 10 countries and each country has one league
season = pd.read_sql("""SELECT season, league.name FROM match INNER JOIN league 
                        ON match.league_id=league.id
                        GROUP BY season, league.name;""", conection)
season # time-span is from 2008/2009 to 2015/2016
season_league = pd.read_sql("""SELECT season,
                                      COUNT(DISTINCT(league_id)) AS season_total_leagues
                               FROM match
                               GROUP BY season;""", conection)
season_league # there are 11 leagues in each season
league_teams = pd.read_sql("""SELECT season,
                                           league.name as league,
                                           COUNT(DISTINCT(home_team_api_id)) AS no_of_home_teams,
                                           COUNT(DISTINCT(away_team_api_id)) AS no_of_away_teams
                                    FROM match
                                    INNER JOIN league ON match.league_id=league.id
                                    GROUP BY season,
                                             league_id;""", conection)
league_teams[league_teams['no_of_home_teams']!=league_teams['no_of_away_teams']] 
conection.execute("""DROP VIEW IF EXISTS team_at_home_win_info""") 
conection.execute("""CREATE TEMP VIEW team_at_home_win_info AS 
                            SELECT season,
                                   league.name as league,
                                   hometeam.team_long_name AS home_team,
                                   country.name AS country,
                                   COUNT(*) as total_matches,
                                   SUM(home_team_goal>away_team_goal) AS team_wins_at_home, 
                                   ROUND((SUM(home_team_goal>away_team_goal)*100.0)/COUNT(*),2) AS team_win_at_home_percentage,
                                   SUM(home_team_goal) as goals_scored_by,
                                   SUM(away_team_goal) as goals_scored_against
                            FROM match 
                            LEFT JOIN team AS hometeam ON match.home_team_api_id=hometeam.team_api_id
                            LEFT JOIN country ON match.country_id=country.id
                            LEFT JOIN league ON match.league_id=league.id 
                            GROUP BY season,
                                     league.name,
                                     country.name,
                                     hometeam.team_long_name
                            ORDER BY team_win_at_home_percentage DESC"""
                 ) 
team_at_home_win = pd.read_sql("""SELECT * FROM team_at_home_win_info""", conection) 
team_at_home_win.head()
team_at_home_win[:20].plot(kind='bar',figsize=(12,6), x='home_team', y='team_win_at_home_percentage', title='Winning % of teams playing home');
team_at_home_win[-20:].plot(kind='bar',figsize=(12,6), x='home_team', y='team_win_at_home_percentage', title='Winning % of teams playing home');
team_win_at_home_percentage = pd.read_sql("""SELECT SUM(home_team_goal>away_team_goal) as team_wins_at_home,
                                                 COUNT(*) as total_matches,
                                                 ROUND((SUM(home_team_goal>away_team_goal)*100.0)/COUNT(*),2) as team_win_at_home_total_percentage
                                          FROM match """, conection) # calculated based on total wins at home by the total no. of matches
team_win_at_home_percentage
conection.execute("""DROP VIEW IF EXISTS team_away_win_info""") 
conection.execute("""CREATE TEMP VIEW team_away_win_info AS 
                            SELECT season,
                                   league.name as league,
                                   awayteam.team_long_name AS away_team,
                                   country.name AS country,
                                   COUNT(*) as total_matches,
                                   SUM(home_team_goal<away_team_goal) AS team_wins_when_away, 
                                   ROUND((SUM(home_team_goal<away_team_goal)*100.0)/COUNT(*),2) AS team_win_when_away_percentage,
                                   SUM(home_team_goal) as goals_scored_by,
                                   SUM(away_team_goal) as goals_scored_against
                            FROM match 
                            LEFT JOIN team AS awayteam ON match.away_team_api_id=awayteam.team_api_id
                            LEFT JOIN country ON match.country_id=country.id
                            LEFT JOIN league ON match.league_id=league.id
                            GROUP BY season,
                                     league.name,
                                     country.name,
                                     awayteam.team_long_name
                            ORDER BY team_win_when_away_percentage DESC"""
                 ) 
team_when_away_win = pd.read_sql("""SELECT * FROM team_away_win_info""", conection) 
team_when_away_win.head()
team_when_away_win[:20].plot(kind='bar',figsize=(12,6), x='away_team', y='team_win_when_away_percentage', title='Winning % of teams when playing away');
team_when_away_win[-20:].plot(kind='bar',figsize=(12,6), x='away_team', y='team_win_when_away_percentage', title='Winning % of teams when playing away');
team_win_when_away_percentage = pd.read_sql("""SELECT SUM(home_team_goal<away_team_goal) as team_wins_when_away,
                                                 COUNT(*) as total_matches,
                                                 ROUND((SUM(home_team_goal<away_team_goal)*100.0)/COUNT(*),2) as team_win_when_away_total_percentage
                                          FROM match """, conection) # calculated based on total wins at home by the total no. of matches
team_win_when_away_percentage
conection.execute("""DROP VIEW IF EXISTS team_performance""") 
conection.execute("""CREATE TEMP VIEW team_performance AS 
                         SELECT team_at_home_win_info.season,
                                team_at_home_win_info.league,
                                team_at_home_win_info.home_team AS team,
                                team_at_home_win_info.country,
                                team_at_home_win_info.team_wins_at_home,
                                team_away_win_info.team_wins_when_away,
                                (team_at_home_win_info.team_wins_at_home + team_away_win_info.team_wins_when_away) AS total_wins,
                                (team_at_home_win_info.goals_scored_by + team_away_win_info.goals_scored_against) AS goals_scored_by,
                                (team_at_home_win_info.goals_scored_against + team_away_win_info.goals_scored_by) AS goals_scored_against,
                                team_at_home_win_info.team_win_at_home_percentage,
                                team_away_win_info.team_win_when_away_percentage 
                         FROM team_at_home_win_info
                         INNER JOIN team_away_win_info ON
                                    team_at_home_win_info.home_team=team_away_win_info.away_team AND 
                                    team_at_home_win_info.league=team_away_win_info.league AND
                                    team_at_home_win_info.season=team_away_win_info.season
                         ORDER BY team_at_home_win_info.season,
                                  team_at_home_win_info.league""")
team_perf = pd.read_sql("""SELECT * FROM team_performance""", conection) 
team_perf.head()
conection.execute("""DROP VIEW IF EXISTS league_team_rank""") 
conection.execute("""CREATE TEMP VIEW league_team_rank AS
                                 SELECT team_performance.season,
                                            league,
                                            team,
                                            team_performance.total_wins,
                                            team_performance.goals_scored_by,
                                            team_performance.goals_scored_against,
                                            RANK() OVER(
                                                        PARTITION BY season, 
                                                                     league
                                                        ORDER BY team_performance.total_wins DESC, 
                                                                 team_performance.goals_scored_by DESC, 
                                                                 team_performance.goals_scored_against 
                                            )team_rank 
                                  FROM team_performance""")
league_team_rank = pd.read_sql("""SELECT * FROM league_team_rank""", conection)
league_team_rank.head()
league_team_rank[league_team_rank['league']=='Germany 1. Bundesliga'].groupby('team_rank').apply(
    lambda x: x["team"].value_counts(ascending=False))
# function to plot competing teams at ith rank (here winning_position) in a league (here league_name)
def plot_ith_position_winner(league_name, winning_position):
    """competing winner teams at winning position in all seasons"""
    league_team_rank[(league_team_rank['league']==league_name) & (league_team_rank['team_rank']==winning_position)]['team'].value_counts().plot(kind='bar', figsize=(8,6), title=f"Winner teams of {league_name} at Position {winning_position}");
"""check competing teams for 1st position in Germany 1. Bundesliga"""
plot_ith_position_winner('Germany 1. Bundesliga', 1)
"""check competing teams for 1st position in Belgium Jupiler League"""
plot_ith_position_winner('Belgium Jupiler League', 1)
conection.execute("""DROP VIEW IF EXISTS league_winner_team""") 
conection.execute("""CREATE TEMP VIEW league_winner_team AS
                                 SELECT season,
                                        league,
                                        team,
                                        total_wins,
                                        goals_scored_by,
                                        goals_scored_against
                                 FROM league_team_rank
                                 WHERE team_rank = 1
                                 ORDER BY season,
                                          league,
                                          team""")
league_winner_team = pd.read_sql("""SELECT * FROM league_winner_team""", conection)
league_winner_team.head()
league_winner_team[league_winner_team['season']=='2008/2009']
sns.pairplot(league_winner_team[['total_wins', 'goals_scored_by']]);
league_winner_team['team'].value_counts(ascending=False).plot(kind='bar', figsize=(18,8));
league_max_goals_team = pd.read_sql("""
                                    SELECT season,
                                           country,
                                           league,
                                           team,
                                           goals_scored_by
                                    FROM
                                    (SELECT season,
                                            country,
                                            league,
                                            team,
                                            goals_scored_by,
                                            RANK() OVER(
                                                        PARTITION BY season, 
                                                                     league
                                                        ORDER BY team_performance.goals_scored_by DESC
                                                        )team_rank_by_goals
                                      FROM team_performance
                                     )teams_performance_season_league
                                     WHERE teams_performance_season_league.team_rank_by_goals = 1  
                                     ORDER BY season,
                                              league,
                                              team
                                 """, conection)
league_max_goals_team.head()
league_max_goals_team.shape
league_max_goals_team.groupby(['season', 'league',]).filter(lambda x: x['team'].count()>1)
league_max_goals_team = pd.read_sql("""
                                    SELECT season,
                                           country,
                                           league,
                                           team,
                                           goals_scored_by
                                    FROM
                                    (SELECT season,
                                            country,
                                            league,
                                            team,
                                            goals_scored_by,
                                            RANK() OVER(
                                                        PARTITION BY season
                                                        ORDER BY team_performance.goals_scored_by DESC
                                                        )team_rank_by_goals
                                     FROM team_performance
                                     )teams_performance_season
                                     WHERE teams_performance_season.team_rank_by_goals = 1  
                                     ORDER BY season
                                 """, conection)
league_max_goals_team
league_max_goals_team['team'].value_counts(ascending=False).plot(kind='bar', figsize=(7,5));
conection.execute("""DROP VIEW IF EXISTS team_ith_win""") 
conection.execute("""CREATE TEMP VIEW team_ith_win AS
                                 SELECT league,
                                        season,
                                        team AS winner_team,
                                        total_wins AS total_wins,
                                        ROW_NUMBER() OVER(
                                        PARTITION BY league,team
                                        ORDER BY season, 
                                                 total_wins DESC
                                        )ith_win
                                 FROM
                                 league_winner_team
                                 ORDER BY league,
                                          season""")
team_ith_win = pd.read_sql("""SELECT * FROM team_ith_win""", conection)
team_ith_win.head()
team_ith_win[team_ith_win['league']=="Belgium Jupiler League"]
team_ith_win[team_ith_win['league']=="France Ligue 1"]
league_winner_team= pd.read_sql("""SELECT league,
                                          max_winning_team,
                                          wins_in_league
                                   FROM
                                   (SELECT league,
                                           winner_team,
                                           FIRST_VALUE(winner_team) OVER(
                                                                         PARTITION BY league
                                                                         ORDER BY ith_win DESC, total_wins DESC
                                           )max_winning_team,
                                           FIRST_VALUE(ith_win) OVER(
                                                                     PARTITION BY league
                                                                     ORDER BY ith_win DESC, total_wins DESC
                                           )wins_in_league
                                     FROM team_ith_win 
                                    )winner_team
                                    GROUP BY league,
                                             max_winning_team,
                                             wins_in_league
                                   """, conection)
league_winner_team
