#Imports



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import sqlite3

import matplotlib.pyplot as plt



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



path = "../input/"  #Insert path here

database = path + 'database.sqlite'
conn = sqlite3.connect(database)



tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)

tables
countries = pd.read_sql("""SELECT *

                        FROM Country;""", conn)

countries
teams = pd.read_sql("""SELECT *

                        FROM Team

                        Where team_long_name ='Tottenham Hotspur';""", conn)

teams
match = pd.read_sql("""SELECT *

                        FROM Match;""", conn)

match
league = pd.read_sql("""SELECT *

                        FROM League;""", conn)

league
leagues = pd.read_sql("""SELECT *

                        FROM League

                        LEFT JOIN Country 

                        ON Country.id = League.country_id;""", conn)

leagues
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

                                Left JOIN Country on Country.id = Match.country_id

                                Left JOIN League on League.id = Match.league_id

                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id

                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id

                                WHERE country_name = 'England'

                                AND season = '2015/2016'

                                ORDER by date

                                LIMIT 10;""", conn)

detailed_matches


tot = pd.read_sql("""SELECT m.date,

t.team_long_name AS 'opponent',

CASE WHEN m.home_team_goal < m.away_team_goal THEN 'Spurs win!'

        WHEN m.home_team_goal > m.away_team_goal THEN 'Spurs loss :(' 

        ELSE 'Tie' END AS outcome

                        FROM match m

                        LEFT JOIN Team t

                        ON m.home_team_api_id = t.team_api_id

                        WHERE m.away_team_api_id = 8586

                        ;""", conn)

tot

         
per = pd.read_sql("""SELECT 

c.name AS country,

ROUND(AVG(CASE WHEN m.season='2013/2014' AND m.home_team_goal = m.away_team_goal THEN 1

 WHEN m.season='2013/2014' AND m.home_team_goal != m.away_team_goal THEN 0

END),2) AS pct_ties_2013_2014,

ROUND(AVG(CASE WHEN m.season='2014/2015' AND m.home_team_goal = m.away_team_goal THEN 1

 WHEN m.season='2014/2015' AND m.home_team_goal != m.away_team_goal THEN 0

END),2) AS pct_ties_2014_2015

FROM country AS c

LEFT JOIN match AS m

ON c.id = m.country_id

GROUP BY country

;""", conn)



per
leages_by_season = pd.read_sql("""



SELECT 

    l.name AS league,

    avg(m.home_team_goal + m.away_team_goal) AS avg_goals

    FROM league as L

    LEFT JOIN match AS m

ON l.id = m.country_id





WHERE m.season = '2013/2014'

    GROUP BY league

    ;""", conn)

leages_by_season

    

leages_by_season = pd.read_sql("""



    SELECT 

    l.name AS league,

    ROUND(avg(m.home_team-goal + m.away_team_goal), 2) AS avg_goals,

    (SELECT ROUND(avg(home_goal + away_goal), 2) 

     FROM match

     WHERE season = '2013/2014') AS overall_avg



FROM 



league AS l

LEFT JOIN match AS m

ON l.id = m.country_id



WHERE m.season = '2013/2014'

GROUP BY league

    ;""", conn)

leages_by_season
leages_by_season = pd.read_sql("""SELECT 

                                        League.name AS league_name, 

                                        season,

                                

                                        avg(home_team_goal) AS avg_home_team_scors, 

                                        avg(away_team_goal) AS avg_away_team_goals, 

                                        avg(home_team_goal-away_team_goal) AS avg_goal_dif, 

                                        avg(home_team_goal+away_team_goal) AS avg_goals, 

                                        sum(home_team_goal+away_team_goal) AS total_goals

                                     

                                FROM Match

                             

                                JOIN League on League.id = Match.league_id

                                LEFT JOIN Team AS HT on HT.team_api_id = Match.home_team_api_id

                                LEFT JOIN Team AS AT on AT.team_api_id = Match.away_team_api_id

                               

                                GROUP BY  League.name, season

                              

                                ;""", conn)

leages_by_season
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
players_height.plot(x=['calc_height'],y=['avg_overall_rating'],figsize=(12,5),title='Potential vs Height')
att = pd.read_sql("""SELECT *

                        FROM Player_Attributes;""", conn)

att