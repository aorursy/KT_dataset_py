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

# Use this as path to input the database file "conn = sqlite3.connect('/kaggle/input/ipldatabase/database.sqlite')"
import sqlite3

import matplotlib.pyplot as plt

import seaborn as sns

#path = "../input/"  #Insert path here

#database = path + 'database.sqlite'



#from subprocess import check_output



conn = sqlite3.connect('/kaggle/input/ipldatabase/database.sqlite')
# to check the table names in the database

tables = pd.read_sql_query("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)

tables
countries = pd.read_sql_query("""SELECT *

                        FROM Country;""", conn)

countries
teams = pd.read_sql("""SELECT *

                        FROM Team

                        """, conn)

teams
# Joining tables Team_id and Player_match with match to get the team performance

team_table = pd.read_sql_query ("""Select m.Match_id,Team_1,Team_2,Team_Name,Match_winner,Win_type,Toss_Decide from Match m 

                                 join Player_Match pm

                                 on pm.match_id = m.match_id

                                 join team t

                                 on t.team_id = pm.team_id""",conn)



team_table.head(20)
# connecting table Player_Match, Player adn Rolee to ccheck the roles of players and their skills only for country =1 i.e India

role = pd.read_sql_query ("""Select pm.Role_Id,Role_Desc,p.Player_Id,Player_Name,Country_Name,Batting_hand,Bowling_skill

                            from Rolee r join Player_match pm

                            on r.Role_Id = pm.Role_Id

                            join Player p

                            on p.Player_Id = pm.Player_Id

                            where Country_Name=1

                            order by DOB desc""",conn)

role.head(20)
seasons = pd.read_sql_query("""Select * from Season order by Season_Year"""

                

                            ,conn)



seasons
ball_by_ball = pd.read_sql_query("""Select bs.Match_Id,Runs_Scored,Team_batting,Team_Bowling 

                                from Ball_by_Ball bb join Batsman_Scored bs

                                on bs.Match_Id = bb.Match_Id""",conn)



ball_by_ball.head()
## Using Group By and having

venue_city = pd.read_sql_query("""Select Venue_Id, Country_Name, count(City_Name) as 'Number of Cities'

                                from Venue v join City ct on ct.City_Id = v.City_Id

                                join Country cy

                                on cy.Country_Id= ct.Country_Id

                                group by Country_Name

                                having Country_Name= "India" or Country_Name='U.A.E'

                                """,conn)



venue_city
metric = pd.read_sql_query("""SELECT 'Matches' As Dimension , COUNT(*) As 'Measure'

                            FROM Match

                   UNION ALL

                   

                   SELECT 'Extra_Runs' As Dimension , SUM(Extra_Runs.Extra_Runs) As 'Measure'

                   FROM Extra_Runs

                   UNION ALL

                   

                   SELECT 'Batsman_runs' As Dimension , SUM(B.Runs_Scored) As 'Value'

                   FROM Batsman_Scored B

                   UNION ALL

                   

                   SELECT 'Wickets' As Dimension , COUNT(*) As 'Measure'

                   FROM Wicket_Taken

                   UNION ALL



                    SELECT 'Sixes' As Dimension , COUNT(*) As 'Measure'

                    FROM Batsman_Scored

                    WHERE Batsman_Scored.Runs_Scored = 6

                    UNION ALL

                    SELECT 'Fours' As Dimension , COUNT(*) As 'Measure'

                    FROM Batsman_Scored

                    WHERE Batsman_Scored.Runs_Scored = 4

                    UNION ALL

                    SELECT 'Singles' As Dimension , COUNT(*) As 'Measure'

                    FROM Batsman_Scored

                    WHERE Batsman_Scored.Runs_Scored = 1""",conn

       )

metric
captain = pd.read_sql( """SELECT C.Player_Name , COUNT(*) As 'Matches_captained'

                        FROM Player_Match A  JOIN Rolee B

                        ON A.Role_Id = B.Role_Id

                        JOIN Player C

                        ON A.Player_Id = C.Player_Id

                        WHERE A.Role_Id = 4 

                        GROUP BY C.Player_Name

                        ORDER BY Matches_captained DESC;""",conn)





captain
#labels = captain["Player_Name"]

plt.figure(figsize=(12,6))

y = captain["Matches_captained"]



x = captain['Player_Name']

plt.bar(x,y,align='center',color='g')



plt.show()