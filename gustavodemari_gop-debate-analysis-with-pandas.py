%matplotlib inline
import sqlite3
import pandas as pd
sql_conn = sqlite3.connect('../input/database.sqlite')
list_of_tables = sql_conn.execute("SELECT * FROM sqlite_master where type='table'")
print(list_of_tables.fetchall())
'''id INTEGER PRIMARY KEY,
candidate TEXT,
candidate_confidence NUMERIC,
relevant_yn TEXT,
relevant_yn_confidence NUMERIC,
sentiment TEXT,
sentiment_confidence NUMERIC,
subject_matter TEXT,
subject_matter_confidence NUMERIC,
candidate_gold TEXT,
name TEXT,
relevant_yn_gold TEXT,
retweet_count INTEGER,
sentiment_gold TEXT,
subject_matter_gold TEXT,
text TEXT,
tweet_coord TEXT,
tweet_created TEXT,
tweet_id INTEGER,
tweet_location TEXT,
user_timezone TEXT'''
mentions_by_location = pd.read_sql("SELECT tweet_location, count(candidate) as mentions from Sentiment group by tweet_location order by 2 DESC", sql_conn)
mentions_by_location.head(10)
query = """SELECT candidate,
        SUM(CASE sentiment WHEN 'Positive' THEN 1 ELSE 0 END) AS positive,
        SUM(CASE sentiment WHEN 'Negative' THEN 1 ELSE 0 END) as negative,
        SUM(CASE sentiment WHEN 'Neutral' THEN 1 ELSE 0 END) AS neutral
        FROM Sentiment GROUP BY candidate ORDER BY 3 DESC,4 DESC"""
sentiment_by_candidate = pd.read_sql(query, sql_conn)
sentiment_by_candidate
sentiment_by_candidate.plot(kind="barh", x="candidate", color=['lime', 'red', 'lightgray'], stacked=True, title="GOP Debate\nMentions sentiment by candidate")
query = """SELECT user_timezone,
        COUNT(candidate) as mentions,
        SUM(CASE sentiment WHEN 'Positive' THEN 1 ELSE 0 END) AS positive,
        SUM(CASE sentiment WHEN 'Negative' THEN 1 ELSE 0 END) as negative,
        SUM(CASE sentiment WHEN 'Neutral' THEN 1 ELSE 0 END) AS neutral
        FROM Sentiment 
        GROUP BY user_timezone ORDER BY 3 DESC,4 DESC"""
sentiment_by_timezone = pd.read_sql(query, sql_conn)
sentiment_by_timezone
query = """SELECT 
        name,
        text,
        retweet_count,
        sentiment
        FROM Sentiment 
        ORDER BY 3 DESC"""
top10_retweet = pd.read_sql(query, sql_conn)
top10_retweet.head(10)