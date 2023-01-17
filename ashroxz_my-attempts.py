# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT *
FROM Teams
LIMIT 5""", con)
print(sample)

# You can read a CSV file like this
users = pd.read_csv("../input/Users.csv")
print(users)

# It's yours to take from here!