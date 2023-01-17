# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full").T
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
hacker_news.head("full").T
stories = '''SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
          '''
types_of_stories = hacker_news.query_to_pandas_safe(stories)
types_of_stories.head()
hacker_news.head('comments').T
deleted_comments = """  
        SELECT COUNT(id) as DEL_COMMENTS
        FROM `bigquery-public-data.hacker_news.comments`
        GROUP BY deleted
        HAVING deleted = TRUE
                    """
num_deleted_comments = hacker_news.query_to_pandas_safe(deleted_comments)
num_deleted_comments
hacker_news.head("full").T
hs_query = """
            SELECT type, count(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            """
high_score = hacker_news.query_to_pandas_safe('hs_query')
high_score

