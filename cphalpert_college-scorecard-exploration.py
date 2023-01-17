import pandas as pd
import seaborn as sns
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

sns.set(style='white')
sns.set_context("poster")

sample = pd.read_sql_query("""
SELECT * FROM Scorecard limit 10""", con)
sample
