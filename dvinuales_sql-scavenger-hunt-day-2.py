# Your code goes here :)

import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query = """SELECT type, COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type            
        """
count_stories = hacker_news.query_to_pandas_safe(query)

count_stories.head()
# How many comments have been deleted? 
# (If a comment was deleted the "deleted" column in the comments table will have the value "True".)
query = """SELECT COUNT(id) as count
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True           
        """

count_comments = hacker_news.query_to_pandas_safe(query)

count_comments.head()

# How many comments have been deleted? 
# Using COUNTIF instead of COUNT as aggregate method
query = """SELECT COUNTIF(deleted = True) as count
            FROM `bigquery-public-data.hacker_news.comments`   
        """

count_comments = hacker_news.query_to_pandas_safe(query)

count_comments.head()