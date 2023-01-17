import time

import sqlite3

import numpy as np

import pandas as pd 



%matplotlib inline

pd.set_option('display.max_rows', 20)
sql_conn = sqlite3.connect('../input/database.sqlite')

df = pd.read_sql("SELECT created_utc, body, subreddit, score FROM May2015 WHERE body NOT NULL LIMIT 1848883", sql_conn)
df = df.assign(len_body = df["body"].str.len())

df  = df.assign(hour_eesti = pd.to_datetime(df["created_utc"], unit="s").dt.tz_localize("UTC").dt.tz_convert("EET").dt.hour)
df
df.info()
print("Suurim skoor: ", df.score.max())

print("Väikseim skoor: ", df.score.min())

print("Keskmine skoor: ", df.score.mean())
df.score.plot.hist(bins = 30, rwidth = 0.9, log = True);
df.plot.scatter("len_body", "score", alpha = 0.2);
df[(df["score"] > 1000) & (df["len_body"] > 3000)]
df["hour_eesti"].value_counts(sort = False).plot.bar();
df.groupby("hour_eesti")["score"].mean().plot.bar();
(df["subreddit"].value_counts()).to_frame()
(df[df["score"] > 1000]["subreddit"].value_counts()).to_frame()
(df.groupby("subreddit")["score"].mean().sort_values(ascending = False)).to_frame()