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
#import package with helper functions
import bq_helper

#create a helper object 
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data", dataset_name="hacker_news")

hacker_news.head("comments")
hacker_news.list_tables()

query = """SELECT parent, COUNT(id)
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY parent
        HAVING COUNT(id) > 10
"""

popular_story = hacker_news.query_to_pandas_safe(query)
popular_story.head()

#HUNT 1: number of story type
hacker_news.head("full")

query2 = """SELECT type, COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
"""
number_of_stories = hacker_news.query_to_pandas_safe(query2)
number_of_stories.head()

#huNT 2 deleted comments

query_deleted = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted    
"""

number_story_deleted = hacker_news.query_to_pandas_safe(query_deleted)
number_story_deleted.head()