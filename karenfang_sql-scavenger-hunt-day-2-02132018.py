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
# Question 1: How many stories (use the "id" column) are there are each type in the full table?

# print the first couple rows of the "full" table
hacker_news.head("full")

query1 = """SELECT type, count(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
stories_by_type = hacker_news.query_to_pandas_safe(query1)

# show the data
stories_by_type

# export the data to a csv file
stories_by_type.to_csv("stories_by_type.csv")
# question 2: How many comments have been deleted?
query2 = """SELECT deleted, count(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = TRUE
         """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
comments_deleted = hacker_news.query_to_pandas_safe(query2)

# show the data
comments_deleted