%matplotlib inline
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

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
pd.read_sql("SELECT * from Sentiment", sql_conn)
mentions_by_location = pd.read_sql("SELECT tweet_location, count(candidate) as mentions from Sentiment group by tweet_location order by 2 DESC", sql_conn)
mentions_by_location.head(10)
query = """SELECT candidate,
        SUM(CASE sentiment WHEN 'Positive' THEN 1 ELSE 0 END) AS positive,
        SUM(CASE sentiment WHEN 'Negative' THEN 1 ELSE 0 END) as negative,
        SUM(CASE sentiment WHEN 'Neutral' THEN 1 ELSE 0 END) AS neutral
        FROM Sentiment 
        GROUP BY candidate 
        ORDER BY 3 DESC,4 DESC"""
sentiment_by_candidate = pd.read_sql(query, sql_conn)
sentiment_by_candidate
sentarray= np.array(sentiment_by_candidate)
sentarray.shape
Trumpsent = sentarray[1,:]

tot=sum(Trumpsent[1:])


labels = 'Positive', 'Negative', 'Neutral'
cats = [609, 1758, 446]
s = np.array(cats)
p = np.asarray((s/tot)*100)
perc = p.round()
colors = ['yellowgreen', 'lightskyblue', 'lightcoral']
explode = (0.1, 0, 0)  # only "explode" the 1st slice (i.e. 'Positve')

plt.pie(perc, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')
plt.title('Will this be the end of America as we know it?...')
fig = plt.figure()
ax = fig.gca()

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
query = """Select subject_matter, candidate, count(*) AS TweetCount
from Sentiment 
where subject_matter != "None of the above" and candidate != ""
group by candidate, subject_matter
order by candidate"""
subject_by_candidate = pd.read_sql(query, sql_conn)
subject_by_candidate
#axes = fig.add_axes([0.1, 0.1, 10, 0.8])
labels = "subject_matter"
BC = subject_by_candidate[0:10]
BC = np.array(BC)
x = BC[:,0]
BC = np.array(BC)
topics = BC[:,0]
x = topics
occur = BC[:,-1]
y = np.array(occur)
l = np.arange(len(x))
y = occur


plt.figure()
plt.scatter(l[1:],y[1:])
plt.xticks( l, x, rotation=80 )
plt.ylabel('# of times topic discussed')
plt.grid()
plt.title('What did Ben Carson Talk about?')

plt.show()
