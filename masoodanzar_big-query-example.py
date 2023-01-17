# BigQuery Data Set URL :  https://www.kaggle.com/hacker-news/hacker-news/data

import bq_helper  

hacker_news = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 

                                       dataset_name = "hacker_news")

hacker_news.list_tables() # print a list of all the tables in the hacker_news dataset
hacker_news.table_schema("full") # print information on all the columns in the "full" table in the hacker_news dataset
hacker_news.head("full") # preview the first couple lines of the "full" table
hacker_news.head("full", selected_columns=("by","score","parent"), num_rows=3)
query = """SELECT score

            FROM `bigquery-public-data.hacker_news.full`

            WHERE type = "job" """



hacker_news.estimate_query_size(query)
# only run this query if it's less than 100 MB

hacker_news.query_to_pandas_safe(query, max_gb_scanned=0.1)
# check out the scores of job postings (if the query is smaller than 1 gig)

job_post_scores = hacker_news.query_to_pandas_safe(query)
# average score for job posts

job_post_scores.score.mean()
# save our dataframe as a .csv 

job_post_scores.to_csv("job_post_scores.csv")
job_post_scores.value_counts()
# Access anothe table = COMMENTS

hacker_news.head("comments")
# Count of Id Group by Parents

query2 = """SELECT parent, COUNT(id)

            FROM `bigquery-public-data.hacker_news.comments`

            GROUP BY parent

            HAVING COUNT(id) > 85

        """

hacker_news.estimate_query_size(query2)
popular_stories = hacker_news.query_to_pandas_safe(query2)

popular_stories
from google.cloud import bigquery

import pandas as pd
query = """

SELECT 

SUM(score) as total_score, author,

SUM(descendants) as total_decendants,

count(id) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

GROUP BY author

ORDER BY total_score DESC

LIMIT 20

"""

hacker_news.estimate_query_size(query)
top_users =  hacker_news.query_to_pandas_safe(query)

top_users.head(3)
# Plot Top 5 Scores

TopTenScore=top_users.sort_values("total_score")[::-1][:5]

TopTenScore.plot(x="author",y=["total_score", "total_decendants", "total_stories"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top users stats').set_ylabel("Score")
query = """

SELECT 

COUNT(*) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

"""

hacker_news.estimate_query_size(query)
count =  hacker_news.query_to_pandas_safe(query)

count.head(3)



print("Total number of stories {}".format(count.total_stories.values[0]))
query = """

SELECT 

SUM(score) as total_score, 

author,

SUM(descendants) as total_decendants,

count(id) as total_stories

FROM `bigquery-public-data.hacker_news.stories`

WHERE author IS NOT NULL

GROUP BY author

ORDER BY total_stories DESC

LIMIT 30

"""

hacker_news.estimate_query_size(query)
most_freq_users = hacker_news.query_to_pandas_safe(query)

# Look at the first 10 top scores

most_freq_users.sort_values("total_stories")[::-1][:10].plot(x="author",y=["total_score", "total_decendants", "total_stories"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top users stats').set_ylabel("Score")
print("Total number of stories {} posted by 30 most active users".format(most_freq_users.total_stories.sum()))

print("Percentage stories posted by 30 most active users : {}%".format(most_freq_users.total_stories.sum()*100/count.total_stories.values[0]))
# Writing query string for our specific need here

query = """

SELECT score, title

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"Y Combinator|YCombinator")

ORDER BY score  DESC

"""



hacker_news.estimate_query_size(query)
#query_job = client.query(query)

#iterator = query_job.result(timeout=30)

HNrows = hacker_news.query_to_pandas_safe(query) #list(iterator)
# Transform the rows into a nice pandas dataframe

#HN_scores = pd.DataFrame(data=[list(x.values()) for x in HNrows], columns=list(HNrows[0].keys()))

HN_scores.sort_values("score")[::-1][:10].plot(x="title",y=["score"], 

                                                   kind="bar",figsize=(14,6),

                                                   title='Top scoring YC Stories').set_ylabel("Score")
print("Total number of stories with mention of YC {}".format(len(HN_scores)))

print("Avg score of stories with mention of YC {}".format(HN_scores.score.mean()))
# Writing query string for our specific need here

query = """

SELECT AVG(score) as avg_score

FROM `bigquery-public-data.hacker_news.stories`

WHERE id NOT IN (SELECT id FROM `bigquery-public-data.hacker_news.stories` WHERE REGEXP_CONTAINS(title, r"Y Combinator|YCombinator")) 

"""

hacker_news.estimate_query_size(query)
Non_HN_score = hacker_news.query_to_pandas_safe(query) 

print("Total average score of stories without mention of HN {}".format(Non_HN_df.avg_score.values[0]))
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(U|u)ber")

GROUP BY year

ORDER BY year

"""
uber_stories_df = hacker_news.query_to_pandas_safe(query) 

uber_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Uber stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score, 

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(H|h)ewlett(-)(P|p)ackard|HP|hp")

GROUP BY year

ORDER BY year

"""
hp_stories_per_year = hacker_news.query_to_pandas_safe(query) 

hp_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='HP stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(T|t)esla")

GROUP BY year

ORDER BY year

"""



tesla_stories_per_year = hacker_news.query_to_pandas_safe(query) 



tesla_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Tesla stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(Y|y)ahoo")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)



yahoo_stories_per_year = hacker_news.query_to_pandas_safe(query) 



yahoo_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Yahoo stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(F|f)acebook")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)



fb_stories_per_year = hacker_news.query_to_pandas_safe(query) 



fb_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Facebook stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(T|t)witch.tv")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
twitch_stories_per_year = hacker_news.query_to_pandas_safe(query) 

twitch_stories_df = pd.DataFrame(data=[list(x.values()) for x in twitch_stories_per_year], columns=list(twitch_stories_per_year[0].keys()))

twitch_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Twitch stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(R|r)eddit")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
reddit_stories_per_year = hacker_news.query_to_pandas_safe(query) 

reddit_stories_df = pd.DataFrame(data=[list(x.values()) for x in reddit_stories_per_year], columns=list(reddit_stories_per_year[0].keys()))

reddit_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Reddit stories per year').set_ylabel("Score")
query ="""

SELECT count(*) as total_stories, SUM(score) as total_score,

EXTRACT(YEAR FROM time_ts) AS year

FROM `bigquery-public-data.hacker_news.stories`

WHERE REGEXP_CONTAINS(title, r"(Z|z)illabyte")

GROUP BY year

ORDER BY year

"""



hacker_news.estimate_query_size(query)
zillabyte_stories_per_year = hacker_news.query_to_pandas_safe(query) 

zillabyte_stories_df = pd.DataFrame(data=[list(x.values()) for x in zillabyte_stories_per_year], columns=list(zillabyte_stories_per_year[0].keys()))

zillabyte_stories_df.plot(x="year",y=["total_stories","total_score"],

                     kind="bar",figsize=(14,6),

                     title='Zillabyte stories per year').set_ylabel("Score")