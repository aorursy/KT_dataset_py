# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")

#
# query to pass to 
query = """SELECT parent, COUNT(id) as comments
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
#This refers to the 'full' table - not the comments table 
hacker_news.head("full")
story_query = """
Select type,COUNT(id) as quantity
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type 
"""


stories_type = hacker_news.query_to_pandas_safe(story_query)

stories_type
#How many comments have been deleted? 
#(If a comment was deleted the "deleted" column in the comments table will have the value "True".)

deleted_query = """

Select deleted,COUNT(id) as quantity
FROM `bigquery-public-data.hacker_news.full`
GROUP BY deleted
"""

deleted = hacker_news.query_to_pandas_safe(deleted_query)
#deleted['quantity'].value_counts(normalize=True)
deleted['percent'] = deleted['quantity']/deleted['quantity'].sum()
deleted['percent'] = deleted['percent'].map(lambda n: '{:,.2%}'.format(n))
deleted
#Other aggregate functions: Count(), sum(), 