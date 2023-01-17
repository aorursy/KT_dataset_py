import sqlite3

import pandas as pd
sql_conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql('SELECT subreddit, count(id) FROM May2015 GROUP BY subreddit ORDER by count(id) DESC LIMIT 50', sql_conn)
df