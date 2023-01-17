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
#How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query_one =  """SELECT type, COUNT(id)
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
             """
type_count = hacker_news.query_to_pandas_safe(query_one)
type_count
#How many comments have been deleted?
query2 = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
         """
deleted_comment_count = hacker_news.query_to_pandas_safe(query2)
deleted_comment_count
query4 = """SELECT `by`, score
                FROM `bigquery-public-data.hacker_news.full`
                WHERE `by` = 'trotterdylan'
             """
score_for_trotterdylan = hacker_news.query_to_pandas_safe(query4)
score_for_trotterdylan
#Optional extra credit: find average of scores per author
query3 =  """SELECT `by`, AVG(score)
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY `by`
                ORDER BY AVG(score) DESC
             """
average_score = hacker_news.query_to_pandas_safe(query3)
average_score.head()