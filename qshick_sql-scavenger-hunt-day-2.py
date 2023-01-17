# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments", 10)
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Let's check how many and what kind of tables we have in the dataset
hacker_news.tables
# 'full' table seems to be something containing many different types as well as 'comments'
hacker_news.head("full", 10)
# Don't forget to use 'id' column to count them
story_query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING type = 'story'
        """
hacker_news.query_to_pandas_safe(story_query)
deleted_query = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """
hacker_news.query_to_pandas_safe(deleted_query)
# DISTINCT is one of aggregation keywords to aggregate cells distinctly
story_distinct_query = """SELECT DISTINCT id
            FROM `bigquery-public-data.hacker_news.full`
            WHERE type = 'story'
        """
# This might take awhile... and will not give you a number but show a list
hacker_news.query_to_pandas_safe(story_distinct_query)
# COUNTIF also counts but you can give it a condition right inside
deleted_countif_query = """SELECT COUNTIF(deleted = True)
            FROM `bigquery-public-data.hacker_news.comments`
        """
# Note that the number is same as found above
hacker_news.query_to_pandas_safe(deleted_countif_query)