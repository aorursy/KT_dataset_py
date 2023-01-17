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
# let's first estimate the size of our query
hacker_news.estimate_query_size(query)
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()
# Your code goes here :)
# lets look at the full table of hacker_news dataset
hacker_news.head("full")
# query for counting comments by their types
query_storytypes = """SELECT type, COUNT(id)
                       FROM `bigquery-public-data.hacker_news.full`
                       GROUP BY type
                   """
# query size estimation
hacker_news.estimate_query_size(query_storytypes)
# safe query
comments_num_by_types = hacker_news.query_to_pandas_safe(query_storytypes)
# the result
comments_num_by_types
# lets look at the data type of column 'deleted' in the full table
hacker_news.table_schema('full')
# query for counting deleted comments
query_num_commentsdeleted = """SELECT COUNT(id)
                                FROM `bigquery-public-data.hacker_news.full`
                                WHERE deleted = True
                            """
# estimation of query size
hacker_news.estimate_query_size(query_num_commentsdeleted)
# running query
del_comments_num = hacker_news.query_to_pandas_safe(query_num_commentsdeleted)
# the result
del_comments_num
# another version of getting previous result
query_num_commentsdeleted2 = """SELECT COUNTIF(deleted = True)
                                 FROM `bigquery-public-data.hacker_news.full`
                             """
del_comments_num2 = hacker_news.query_to_pandas_safe(query_num_commentsdeleted)

del_comments_num2