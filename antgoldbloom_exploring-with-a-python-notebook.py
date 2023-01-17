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
LIMIT 10
""", con))

top10 = pd.read_sql_query("""
SELECT *
FROM Users
WHERE Ranking IS NOT NULL
ORDER BY Ranking
LIMIT 10
""", con)
print(top10)

print(pd.read_sql_query("""
SELECT *
FROM Users
WHERE HighestRanking=1
""", con))
import matplotlib
matplotlib.style.use('ggplot')

top10.sort(columns="Ranking").plot(x="DisplayName", y="Ranking", kind="barh", color="#20beff")
