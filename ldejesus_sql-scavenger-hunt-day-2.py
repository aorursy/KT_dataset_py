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
# Stories in full table
query_count_stories = """SELECT type, COUNT(id)
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
count_types = hacker_news.query_to_pandas_safe(query_count_stories)

count_types.head()

#comments deleted
query_comments_deleted = """SELECT deleted, COUNT(id)
    FROM `bigquery-public-data.hacker_news.comments`
    GROUP BY deleted
    HAVING deleted
        """
comments_deleted = hacker_news.query_to_pandas_safe(query_comments_deleted)

comments_deleted.head()

#SUM comments
query_comments_sum = """SELECT deleted, SUM(id)
    FROM `bigquery-public-data.hacker_news.comments`
    GROUP BY deleted
    HAVING deleted
        """
comments_sum = hacker_news.query_to_pandas_safe(query_comments_sum)

comments_sum.head()
