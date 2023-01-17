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

# check the table list for the dataset
hacker_news.list_tables()

# print the structure of the "comments" table
hacker_news.table_schema("comments")

# print the structure of the "full" table
hacker_news.table_schema("full")

# print few records from "full" table
hacker_news.head("full")

# below query would return count of each type from the "full" table
query = """SELECT type,COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type """

# check how big this query will be
hacker_news.estimate_query_size(query)

# require 253 MB to run the query
unique_stories = hacker_news.query_to_pandas(query)

# count of unique stories
unique_stories.type.count()

# question 2 of hunt, inspect "comments" table of Hacker News dataset
hacker_news.list_tables()
hacker_news.table_schema("comments")
hacker_news.head("comments")

# below query will bring count of all the deleted comments
# as the "id" coulmn nullable we should take simple COUNT to include NULL
query = """SELECT COUNT(*) AS cnt
             FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True """

# estimate size of the query
hacker_news.estimate_query_size(query)

# print the number of deleted comments
del_cmts = hacker_news.query_to_pandas(query)

# optional extra credit question answer
query = """SELECT COUNTIF(deleted = True) AS cnt
             FROM `bigquery-public-data.hacker_news.comments` """

# estimate query size
hacker_news.estimate_query_size(query)

#only needed 0.2MB, much useful than the earlier query
del_cmts_2 = hacker_news.query_to_pandas(query)

#print out the number of deleted comments
del_cmts.cnt.count()