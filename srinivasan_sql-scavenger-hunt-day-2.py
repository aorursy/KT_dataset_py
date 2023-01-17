# SQL Scavenger Hunt Day #2 -
# How many stories (use the "id" column) are there of each type (in the "type" column) 
#in the full table 

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# query to pass to 
query = """SELECT type, COUNT(*) 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

stories_and_their_types = hacker_news.query_to_pandas_safe(query)

print(stories_and_their_types)

# SQL Scavenger Hunt Day #2 
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)

# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# query to pass to 
query = """SELECT COUNT(*) 
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """

deleted_comment_count = hacker_news.query_to_pandas_safe(query)

print(deleted_comment_count)
# modify one of the queries you wrote above to use a different aggregate function