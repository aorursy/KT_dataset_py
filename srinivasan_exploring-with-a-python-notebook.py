%matplotlib inline
import pandas as pd
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
print(pd.read_sql_query("""
SELECT c.CompetitionName,
       COUNT(t.Id) NumberOfTeams
FROM Competitions c
INNER JOIN Teams t ON t.CompetitionId=c.Id
-- ONLY including teams that ranked
WHERE t.Ranking IS NOT NULL
GROUP BY c.CompetitionName
ORDER BY COUNT(t.Id) DESC
LIMIT 5
""", con))
top5 = pd.read_sql_query("""
SELECT *
FROM Users
WHERE Ranking IS NOT NULL
ORDER BY Ranking, RegisterDate 
LIMIT 5
""", con)
print(top5)
print(pd.read_sql_query("""
SELECT *
FROM Users
WHERE HighestRanking=1
""", con))
import matplotlib
matplotlib.style.use('ggplot')

top5.sort(columns="Points").plot(x="DisplayName", y="Points", kind="barh", color="#20beff")
