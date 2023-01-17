# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import sqlite3



path = "../input/"  #Insert path here

database = path + 'database.sqlite'
conn = sqlite3.connect(database)



tables = pd.read_sql("""Select * from sqlite_master where type = 'table';""",conn)

tables
country = pd.read_sql("""select * from Country;""",conn)
country
leagues = pd.read_sql("""select * from League INNER JOIN Country ON League.id = Country.id;""",conn)

leagues   #List of leagues and their Country
#List of teams



teams = pd.read_sql("""select * 

                          from Team 

                          Order by team_long_name

                          Limit 10;""",conn)

teams
#List of matches 



detailed_matches = pd.read_sql("""SELECT Match.id, 

                                        Country.name AS country_name, 

                                        League.name AS league_name, 

                                        season, 

                                        stage, 

                                        date,

                                        HT.team_long_name AS  home_team,

                                        AT.team_long_name AS away_team,

                                        home_team_goal, 

                                        away_team_goal                                        

                                FROM Match

                                JOIN Country on Country.id = Match.country_id

                                JOIN League on League.id = Match.league_id

                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id

                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id

                                WHERE country_name = 'Spain'

                                ORDER by date

                                LIMIT 10;""", conn)

detailed_matches
leages_by_season = pd.read_sql("""SELECT Country.name AS country_name, 

                                        League.name AS league_name, 

                                        season,

                                        count(distinct stage) AS number_of_stages,

                                        count(distinct HT.team_long_name) AS number_of_teams,

                                        avg(home_team_goal) AS avg_home_team_scors, 

                                        avg(away_team_goal) AS avg_away_team_goals, 

                                        avg(home_team_goal-away_team_goal) AS avg_goal_dif, 

                                        avg(home_team_goal+away_team_goal) AS avg_goals, 

                                        sum(home_team_goal+away_team_goal) AS total_goals                                       

                                FROM Match

                                JOIN Country on Country.id = Match.country_id

                                JOIN League on League.id = Match.league_id

                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id

                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id

                                WHERE country_name in ('Spain', 'Germany', 'France', 'Italy', 'England')

                                GROUP BY Country.name, League.name, season

                                HAVING count(distinct stage) > 10

                                ORDER BY Country.name, League.name, season DESC

                                ;""", conn)

leages_by_season
df = pd.DataFrame(index=np.sort(leages_by_season['season'].unique()), columns=leages_by_season['country_name'].unique())



df.loc[:,'Germany'] = list(leages_by_season.loc[leages_by_season['country_name']=='Germany','avg_goals'])

df.loc[:,'Spain']   = list(leages_by_season.loc[leages_by_season['country_name']=='Spain','avg_goals'])

df.loc[:,'France']   = list(leages_by_season.loc[leages_by_season['country_name']=='France','avg_goals'])

df.loc[:,'Italy']   = list(leages_by_season.loc[leages_by_season['country_name']=='Italy','avg_goals'])

df.loc[:,'England']   = list(leages_by_season.loc[leages_by_season['country_name']=='England','avg_goals'])



df.plot(figsize=(12,5),title='Average Goals per Game Over Time')
df = pd.DataFrame(index=np.sort(leages_by_season['season'].unique()), columns=leages_by_season['country_name'].unique())



df.loc[:,'Germany'] = list(leages_by_season.loc[leages_by_season['country_name']=='Germany','avg_goal_dif'])

df.loc[:,'Spain']   = list(leages_by_season.loc[leages_by_season['country_name']=='Spain','avg_goal_dif'])

df.loc[:,'France']   = list(leages_by_season.loc[leages_by_season['country_name']=='France','avg_goal_dif'])

df.loc[:,'Italy']   = list(leages_by_season.loc[leages_by_season['country_name']=='Italy','avg_goal_dif'])

df.loc[:,'England']   = list(leages_by_season.loc[leages_by_season['country_name']=='England','avg_goal_dif'])



df.plot(figsize=(12,5),title='Average Goals Difference Home vs Out')




players_height = pd.read_sql("""SELECT CASE

                                        WHEN ROUND(height)<165 then 165

                                        WHEN ROUND(height)>195 then 195

                                        ELSE ROUND(height)

                                        END AS calc_height, 

                                        COUNT(height) AS distribution, 

                                        (avg(PA_Grouped.avg_overall_rating)) AS avg_overall_rating,

                                        (avg(PA_Grouped.avg_potential)) AS avg_potential,

                                        AVG(weight) AS avg_weight 

                            FROM PLAYER

                            LEFT JOIN (SELECT Player_Attributes.player_api_id, 

                                        avg(Player_Attributes.overall_rating) AS avg_overall_rating,

                                        avg(Player_Attributes.potential) AS avg_potential  

                                        FROM Player_Attributes

                                        GROUP BY Player_Attributes.player_api_id) 

                                        AS PA_Grouped ON PLAYER.player_api_id = PA_Grouped.player_api_id

                            GROUP BY calc_height

                            ORDER BY calc_height

                                ;""", conn)

players_height


