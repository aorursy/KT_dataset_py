# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

# print a list of all the tables in the hacker_news dataset
hacker_news.list_tables()
# print information on all the columns in the "full" table
# in the hacker_news dataset
hacker_news.table_schema("full")
hacker_news.head("full")
# query to pass to 
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
# popular_stories = hacker_news.query_to_pandas_safe(query)
#popular_stories.head()
# Query to Count entries in each type from "full" table
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        
        """

# Count entries in each type from "full" table
story_types = hacker_news.query_to_pandas_safe(query)
story_types.head()
# Query to Count entries that were deleted
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
        """

# Count entries that were deleted
deleted = hacker_news.query_to_pandas_safe(query)
deleted.head()
# Use AVG aggregate function to show average ranking of entries by author
# Could also try VARIANCE, MAX, MIN
# query = """SELECT author, MAX(ranking)
query = """SELECT author, AVG(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
        """

avg_author_ranking = hacker_news.query_to_pandas_safe(query)

avg_author_ranking