import pandas as pd

import sqlite3



# Read in SQLite databases

con = sqlite3.connect("../input/database.sqlite")

results = pd.read_sql_query("SELECT * from resultsdata13", con)

samples = pd.read_sql_query("SELECT * from sampledata13", con)



con.close()
samples.head()
squash = samples.loc[samples['commod'] == 'SS']

squash.head()
squash['variety'].value_counts()
squash['state'].value_counts()