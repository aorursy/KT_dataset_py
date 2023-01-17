import sqlite3
import pandas as pd



sql_conn = sqlite3.connect('../input/database.sqlite')


query = "SELECT * \
         FROM May2015 \
         WHERE subreddit = 'nba' AND body != '[deleted]'"

df = pd.read_sql(query, sql_conn)
df['body']
