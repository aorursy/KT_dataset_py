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
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
###### Question 1
hacker_news.head("full")
query = """
        SELECT type, count(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """
types_comment_cnt = hacker_news.query_to_pandas_safe(query)
types_comment_cnt.head()


hacker_news.head("full")
#####Question 2
hacker_news.head("full")
query_2="""
      SELECT type, Count(id)
      FROM `bigquery-public-data.hacker_news.full`
      WHERE deleted = True and type = "comment"
      group by type
      """
hacker_news.estimate_query_size(query_2)
no_deleted_comments = hacker_news.query_to_pandas_safe(query_2)
no_deleted_comments.head()
#Bonus credit
hacker_news.head("full")
query_3="""
      SELECT type, COUNTIF(deleted = True)
      FROM `bigquery-public-data.hacker_news.full`
      WHERE type="comment"
      group by type
      """
hacker_news.estimate_query_size(query_3)
no_deleted_comments = hacker_news.query_to_pandas_safe(query_3)
no_deleted_comments.head()