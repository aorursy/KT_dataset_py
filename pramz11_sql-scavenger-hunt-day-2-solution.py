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
# Scavenger Hunt Day 2. Q1. How many stories (use the "id" column) 
# are there of each type (in the "type" column) in the full table?

query = """ SELECT type, COUNT (id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type
             
        """
# check query size before executing 

hacker_news.estimate_query_size(query)

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
story_type = hacker_news.query_to_pandas_safe(query)

# Results
story_type
# Q2. How many comments have been deleted? (If a comment was deleted the "deleted" column in 
# the comments table will have the value "True".)

query = """ SELECT COUNT(id) AS Deleted_comments
             FROM `bigquery-public-data.hacker_news.comments`
             WHERE deleted = TRUE
        """

# estimate query size
hacker_news.estimate_query_size(query)

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query)

# Results
deleted_comments
# Bonus question 
# Query to count average id and group the data set by parent id, use id>10 to filter less popular comments

query = """ SELECT type, AVG (id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type
             
        """
# check query size before executing 

hacker_news.estimate_query_size(query)

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
avg_story_type= hacker_news.query_to_pandas_safe(query)

avg_story_type