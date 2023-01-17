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
# query to pass to 
queryStories = """SELECT type, COUNT(id)
                  FROM `bigquery-public-data.hacker_news.full`
                  GROUP BY type
               """
# saving query run in safe mode to dataframe
story_types = hacker_news.query_to_pandas_safe(queryStories)
story_types
# query to pass to 
queryComments = """SELECT COUNT(id)
                   FROM `bigquery-public-data.hacker_news.comments`
                   GROUP BY deleted
                   HAVING deleted = True
                """
# saving query run in safe mode to dataframe
deleted_comments = hacker_news.query_to_pandas_safe(queryComments)
deleted_comments
# query to find the max ranking for deleted and non-deleted comments
queryMaxComments = """SELECT MAX(ranking)
                      FROM `bigquery-public-data.hacker_news.comments`
                      GROUP BY deleted
                   """
# saving query run in safe mode to dataframe
max_comments = hacker_news.query_to_pandas_safe(queryMaxComments)
max_comments