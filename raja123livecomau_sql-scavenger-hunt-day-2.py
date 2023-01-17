# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
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
# Your code goes here :)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE TYPE <> ' '
            GROUP BY type
            HAVING COUNT(id) > 0
        """
popular_stories = hacker_news.query_to_pandas_safe(query)
display(popular_stories)
# Now, for the second SQL :-)

query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted
            GROUP BY deleted
            HAVING COUNT(id) > 0
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
display(deleted_comments)
