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
# query to pass to find out:
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """SELECT `bigquery-public-data.hacker_news.full`.type, COUNT(`bigquery-public-data.hacker_news.stories`.id)
            FROM `bigquery-public-data.hacker_news.stories`
                join `bigquery-public-data.hacker_news.full`
                ON `bigquery-public-data.hacker_news.stories`.by = `bigquery-public-data.hacker_news.full`.by
            GROUP BY `bigquery-public-data.hacker_news.full`.type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
num_stories_type = hacker_news.query_to_pandas_safe(query)

num_stories_type.head()
# query to pass to find out: How many comments have been deleted?
query = """
        select count(*)
        from `bigquery-public-data.hacker_news.comments`
        where deleted is True
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
num_stories_type = hacker_news.query_to_pandas_safe(query)

num_stories_type.head()