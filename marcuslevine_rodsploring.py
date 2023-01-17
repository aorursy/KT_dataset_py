# The data comes both as CSV files and a SQLite database

import pandas as pd
import sqlite3

# You can read in the SQLite datbase like this
import sqlite3
con = sqlite3.connect('../input/database.sqlite')
sample = pd.read_sql_query("""
SELECT p.Name Sender,
       e.MetadataSubject Subject
FROM Emails e
INNER JOIN Persons p ON e.SenderPersonId=p.Id
LIMIT 10""", con)
print(sample)

# You can read a CSV file like this
persons = pd.read_csv("../input/Persons.csv")
print(persons)

# It's yours to take from here!
pd.read_sql_querya(""" .tables """, con)
pd.read_sql_query('select * from Persons')
