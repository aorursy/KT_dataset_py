import pandas as pd

import sqlite3



# Read in SQLite databases

con = sqlite3.connect("../input/database.sqlite")

results = pd.read_sql_query("SELECT * from resultsdata15", con)

samples = pd.read_sql_query("SELECT * from sampledata15", con)



con.close()
samples.head()
samples.tail()
samples = samples[['state', 'commod', 'variety']]

samples.head()
apples = samples.loc[samples['commod'] == 'AP']
apples.head()
apples['state'].value_counts()
apples['variety'].value_counts()