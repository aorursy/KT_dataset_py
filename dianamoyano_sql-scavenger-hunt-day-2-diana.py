# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")


#1. How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query1=""" SELECT type, COUNT(id)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""
comment_group=hacker_news.query_to_pandas(query1)

comment_group
#2. How many comments have been deleted? (If a comment was deleted the "deleted" column in the comments 
#table will have the value "True".)

query2=""" SELECT deleted, COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
GROUP BY deleted
HAVING deleted=true
"""
com_del=hacker_news.query_to_pandas(query2)

com_del
#Opt. What is the date of the most recent poll?
query3=""" SELECT type, MAX(timestamp)
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
HAVING type="pollopt"
"""
recent_poll=hacker_news.query_to_pandas(query3)

recent_poll