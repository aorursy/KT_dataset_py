import time

import sqlite3

import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)
sql_conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql("SELECT created_utc, body, author, score FROM May2015 WHERE body NOT NULL ORDER BY score DESC LIMIT 5000", sql_conn)
df = df.assign(len_body=df['body'].str.len())

df.created_utc = pd.to_datetime(df.created_utc,unit='s')

df.created_utc = df.created_utc.dt.tz_localize('UTC').dt.tz_convert('EET').dt.hour
df
df.info()
print("Maksimaalne skoor: ", df.score.max())

print("Minimaalne skoor: ", df.score.min())

print("Keskmine skoor: ", df.score.mean())
df.score.plot.hist();
df.plot.scatter("score", "len_body", alpha=0.5);
df.plot.scatter("created_utc", "score", alpha=0.5);
df_tunnid = df.groupby("created_utc")["score"].mean()
df_tunnid.reset_index()