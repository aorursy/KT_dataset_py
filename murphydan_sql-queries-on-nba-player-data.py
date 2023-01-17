import pandas as pd

import pandasql as ps
Players = pd.read_csv('../input/nba-players-stats/Players.csv')

SeasonsStats = pd.read_csv('../input/nba-players-stats/Seasons_Stats.csv')

PlayerData = pd.read_csv('../input/nba-players-stats/player_data.csv')
# First, show a preview of all three tables



ps.sqldf(""" SELECT * 

             FROM Players 

             ORDER BY born DESC 

             LIMIT 5 

         """)
ps.sqldf(""" SELECT * 

             FROM SeasonsStats 

             ORDER BY Year DESC 

             LIMIT 5 

         """)
ps.sqldf(""" 

            SELECT * 

            FROM PlayerData 

            ORDER BY year_end DESC 

            LIMIT 5

         """)
# View all players from Providence College



ps.sqldf(""" 

            SELECT * 

            FROM PlayerData 

            WHERE college = "Providence College"

            ORDER BY year_end DESC 

         """)
# Which PC Players had the longest NBA tenure? 



ps.sqldf(""" 

            SELECT name, year_start, year_end, position, year_end - year_start AS Tenure

            FROM PlayerData 

            WHERE college = "Providence College"

            ORDER BY Tenure DESC 

            LIMIT 7

         """)
# NBA Points Leader from PC



ps.sqldf("""

            SELECT name, SUM(s.PTS) AS CareerPoints

            FROM PlayerData 

            LEFT JOIN SeasonsStats AS s

            ON name = s.Player

            WHERE college = "Providence College"

            GROUP BY name

            ORDER BY CareerPoints DESC 

         """)





# The Lenny Wilkens number doesn't seem right - his seasons seem to be omitted  



ps.sqldf("""

            SELECT Player, Year, Pts

            FROM SeasonsStats 

            WHERE  Player = "Lenny Wilkens"

        """)

# Looking at 1965 as an example, Lenny (and Jerry West) appear with an asterisk. 

# On Basketball Reference where this data was scraped, asterisks indicate the all star years which is affecting the join.  

# https://www.basketball-reference.com/players/w/wilkele01.html



ps.sqldf("""

            SELECT Player, Pts

            FROM SeasonsStats

            WHERE Year = 1965

         """)
# Lets try the point list again after removing the asterisk with a subquery - Now Lenny's numbers look right.  



ps.sqldf("""

             SELECT p.name, SUM(s.PTS) AS CareerPoints

            FROM PlayerData as p

            LEFT JOIN 

                ( SELECT REPLACE(Player, '*', '') AS Player, Pts

                  FROM SeasonsStats

                ) AS s

            ON p.name = s.Player 

            WHERE college = "Providence College"

            GROUP BY name

            ORDER BY CareerPoints DESC 

         """)



# Which friar had the most points during his Rookie Season?  Ernie D was a force before his career was cut short by injuries.  



ps.sqldf("""

            SELECT p.name, p.year_start as Year, s.PTS AS RookieYrPts

            FROM PlayerData as p

            LEFT JOIN 

                ( SELECT REPLACE(Player, '*', '') AS Player, Year, Pts

                  FROM SeasonsStats

                ) AS s

            ON p.name = s.Player 

            AND p.year_start = s.Year

            WHERE college = "Providence College"

            ORDER BY RookieYrPts DESC 

         """)

# What was the most points scored in a season for each Friar? 



ps.sqldf("""

            SELECT REPLACE(s.Player, '*', '') AS Player, s.Year, MAX(s.Pts) AS HighestSeasonPts

            FROM SeasonsStats AS s

            LEFT JOIN 

                ( SELECT name, college

                  FROM PlayerData

                ) AS p

            ON s.Player = p.name 

            WHERE p.college = "Providence College"

            GROUP BY Player

            ORDER BY HighestSeasonPts DESC

         """)





# Show each PC player's seasons ranked by Points. 



# Subquery groups SeasonsStats by year to account for instances where a player was traded.  

# Team = "TOT" is filtered out.  This row sums results when player played on two teams.   



ps.sqldf("""



         SELECT REPLACE(s.Player, '*', '') AS Player, s.Year, SUM(s.Pts) AS SeasonPts, 

           ROW_NUMBER() OVER(PARTITION BY Player ORDER BY SUM(s.Pts) DESC) AS "PlayerSeasonRank"

         FROM SeasonsStats AS s

         LEFT JOIN 

           ( 

             SELECT name, college

             FROM PlayerData

           ) AS p

         ON s.Player = p.name 

         WHERE p.college = "Providence College"

         AND s.Tm != "TOT"

         GROUP BY s.Year, Player 

         """).head(60)            #head() used to force more rows to appear