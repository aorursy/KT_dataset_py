# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# We check the tables that are in the database
hacker_news.list_tables()
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
hacker_news.head("full")
# Your code goes here :)
# Query 1:
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?

# First we write the query:
query_1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# Then we run it, saving it in a pandas dataframe:
number_stories = hacker_news.query_to_pandas_safe(query_1)

# We verify our results:
print(number_stories)
# Your code goes here :)
# Query 2
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# First we write the query:

# First try, without using HAVING 
# query_2 = """SELECT type, deleted, COUNT(id)
#            FROM `bigquery-public-data.hacker_news.full`
#            GROUP BY type, deleted
#        """
# Comments deleted = True : 368371

# Second try, using HAVING
query_2 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type, deleted
            HAVING deleted = True
        """
# Comments deleted = True : 368371, we get the same results

# Then we run it, saving it in a pandas dataframe:
deleted_comments = hacker_news.query_to_pandas_safe(query_2)

# We verify our results:
print(deleted_comments)
# Your code goes here :)
# Query 3
# read about [aggregate functions other than COUNT()]
# (https://cloud.google.com/bigquery/docs/reference/standard-sql/functions-and-operators#aggregate-functions) 
# and modify one of the queries you wrote above to use a different aggregate function.

# First we write the query:
query_3 = """SELECT type, MAX(descendants)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# Then we run it, saving it in a pandas dataframe:
max_descendants = hacker_news.query_to_pandas_safe(query_3)

# We verify our results:
print(max_descendants)