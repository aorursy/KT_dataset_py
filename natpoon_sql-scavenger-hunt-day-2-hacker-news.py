# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# Look at all the tables in the hacker_news dataset
hacker_news.list_tables()
# query to find how many comments each parent comment has 
# (all parent comments with less than 10 comments are excluded)
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
# Before writing the SQL query, I prefer to look at the table schema
# to get a sense of the other columns and whether or not they could be useful.
hacker_news.table_schema("full")
# Your code goes here :)
# Define the query to answer Question #1.
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type"""

# Create a new dataframe based on query (but only if it is less than 1GB)
stories_per_type = hacker_news.query_to_pandas_safe(query)

# View the results.
# Note: I used `.head()` in case there were many types
stories_per_type.head(10)
# Before writing the SQL query, I prefer to look at the table schema
# to get a sense of the other columns and whether or not they could be useful.
hacker_news.table_schema("comments")
# Define the query to answer Question #2.
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = TRUE
        """
# Create a new dataframe based on query (but only if it is less than 1GB)
del_comments = hacker_news.query_to_pandas_safe(query)

# View the results.
del_comments
# Define the query to answer the Extra Credit question.
query = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING AVG(score) > 0
        """

# Create a new dataframe based on query (but only if it is less than 1GB)
avg_type_score = hacker_news.query_to_pandas_safe(query)

# View the results.
avg_type_score