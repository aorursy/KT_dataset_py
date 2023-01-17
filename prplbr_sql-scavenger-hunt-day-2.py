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
hacker_news.list_tables()

query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_stories = hacker_news.query_to_pandas_safe(query1,max_gb_scanned=0.3)
type_stories.head(10)
hacker_news.head("comments")
query2a = """SELECT  COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
        """
total_comments = hacker_news.query_to_pandas_safe(query2a,max_gb_scanned=0.1)
total_comments.head(5)
query2b = """SELECT  COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is TRUE
        """
deleted_comments = hacker_news.query_to_pandas_safe(query2b,max_gb_scanned=0.1)
deleted_comments.head(5)
query2c = """SELECT  deleted,COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
comments_by_delete = hacker_news.query_to_pandas_safe(query2c,max_gb_scanned=0.1)
comments_by_delete.head(5)
hacker_news.head("full")
query3a = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
avg_score_per_type = hacker_news.query_to_pandas_safe(query3a,max_gb_scanned=0.2)
avg_score_per_type.head()

query3b = """SELECT by, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY by
        """
avg_score_per_user = hacker_news.query_to_pandas_safe(query3b,max_gb_scanned=0.2)
avg_score_per_user.head()
query3c = """SELECT `by`, AVG(score), COUNT(score), SUM(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY `by`
            HAVING AVG(score) is not NULL
        """
avg_score_per_user = hacker_news.query_to_pandas_safe(query3c,max_gb_scanned=0.2)

avg_score_per_user.sort_values(by='f2_',ascending=False).head(20)