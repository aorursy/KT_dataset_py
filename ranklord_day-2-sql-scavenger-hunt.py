# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper("bigquery-public-data","hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# query to pass to 
query = """ SELECT parent, COUNT(id) as qty
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
            ORDER BY qty DESC
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head().style.set_properties(**{'text-align': 'right'})
# How many stories are there of each type in the full table?
query = """ SELECT type, COUNT(id) as quantity
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

type_stories = hacker_news.query_to_pandas_safe(query)
type_stories.style.set_properties(**{'text-align': 'right'})
# How many comments have been deleted?
query = """ SELECT COUNT(deleted) as Comments_Deleted
            FROM `bigquery-public-data.hacker_news.comments`
        """

deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.style.set_properties(**{'text-align': 'right'})
# (sligthly) alternat aggregation function
query = """ SELECT COUNTIF(deleted = True) as Comments_Deleted
            FROM `bigquery-public-data.hacker_news.comments`
        """

alternative_agg = hacker_news.query_to_pandas_safe(query)
alternative_agg.style.set_properties(**{'text-align': 'right'})