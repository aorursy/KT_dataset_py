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
# Checking full table on Hacker Data to get a sense on type column
hacker_news.head('full')
# Setting the QUERYs
# Note the COUNT() AS statement used to change the name of the column returned
# Note the ORDER BY DESC statement used to order the result query

QUERY1 = """SELECT type, COUNT(id) AS type_counts
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY COUNT(id) DESC
            """

QUERY2 = """SELECT deleted, COUNT(id) AS deleted_comments_count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            """

QUERY3 = """SELECT deleted, COUNTIF(deleted = True) AS deleted_comments_count
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING COUNTIF(deleted = True) != 0
            """
# Solving question 1
type_count = hacker_news.query_to_pandas_safe(QUERY1)
# Displaying type_counts.head()
type_count.head(10)
# Solving question 2
deleted_comments_count = hacker_news.query_to_pandas_safe(QUERY2)
# Displaying deleted_comments_count.head()
deleted_comments_count.head()
# Solving extra credit
extra_credit = hacker_news.query_to_pandas_safe(QUERY3)
# Displaying first line of extra_credit
extra_credit.head()