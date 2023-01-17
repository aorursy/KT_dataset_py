import numpy as np 

import pandas as pd

import sqlite3



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
path = "../input/"  #Insert path here

database = path + 'database.sqlite'



conn = sqlite3.connect(database)



tables = pd.read_sql("""SELECT *

                        FROM sqlite_master

                        WHERE type='table';""", conn)

tables
query = '''

WITH LatestSubmission AS (

   SELECT MAX(DateSubmitted) AS LastSubmissionDate  -- Get the day of the last recorded submission (i.e. last submission of a completed competition)

   FROM Submissions

),

TotalSubmissionsByUser AS (

    SELECT SubmittedUserId, COUNT(*) As TotalSubmissions

      FROM Submissions

  GROUP BY SubmittedUserId

),

UserDaysRegistered AS (

    SELECT u.Id,

           u.DisplayName,

           u.Ranking,

           (julianday(ls.LastSubmissionDate) - julianday(u.RegisterDate)) AS DaysRegistered

      FROM Users u, LatestSubmission ls

),

TotalDailySubmissionsByUser AS (

    SELECT (ts.TotalSubmissions / udr.DaysRegistered) AS DailySubmissions,

           udr.DisplayName, 

           udr.Ranking,

           udr.DaysRegistered

  FROM UserDaysRegistered udr

  JOIN TotalSubmissionsByUser ts ON  ts.SubmittedUserId = udr.Id

)

SELECT * 

 FROM TotalDailySubmissionsByUser

WHERE DaysRegistered > 100 -- Remove some noise by not including new users

ORDER BY DailySubmissions DESC

LIMIT 100

'''
top100bySubs = pd.read_sql( query, conn )
top100bySubs
top100bySubs.DisplayName.values
top100bySubs.to_csv("top100bySubs.csv", index=False)