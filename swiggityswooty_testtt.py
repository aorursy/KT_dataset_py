# The data comes as the raw data files, a transformed CSV file, and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT INSTNM,
       COSTT4_A AverageCostOfAttendance,
       Year
FROM Scorecard
WHERE INSTNM='Duke University'""", con)
print(sample)

# You can read a CSV file like this
scorecard = pd.read_csv("../input/Scorecard.csv")
print(scorecard)

# It's yours to take from here!