# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query_1 = """SELECT type, COUNT(id) as cnt
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
          """
stoires_count = hacker_news.query_to_pandas_safe(query_1)
stoires_count

query_2 = """SELECT deleted, COUNT(id) as cnt
                FROM `bigquery-public-data.hacker_news.comments`
                WHERE deleted = True
                GROUP BY deleted
          """
delected_comments = hacker_news.query_to_pandas_safe(query_2)
delected_comments
# what's the total score for book in stories table of which the auther is dead?
query_3 = """SELECT dead, sum(score) as total
                FROM `bigquery-public-data.hacker_news.stories`
                WHERE dead = True
                GROUP BY dead
          """
dead_score = hacker_news.query_to_pandas_safe(query_3)
dead_score