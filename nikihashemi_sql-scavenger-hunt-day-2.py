#Part 1: How many stories are there of each type in the full table?
import bq_helper
hacker_news_stories = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query1 = """
            SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
         """
stories_by_type = hacker_news_stories.query_to_pandas_safe(query1)
stories_by_type.head()

#Part 2: How many comments have been deleted?
import bq_helper

hacker_news_comments = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query2 = """
            SELECT deleted, count(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
         """

deleted_comments = hacker_news_comments.query_to_pandas_safe(query2)
deleted_comments.head()
#Part 3: extra credit using MIN and MAX
import bq_helper

hacker_news_comments_edit = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query3 = """
            SELECT deleted, min(id), max(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
         """

deleted_comments_edit = hacker_news_comments_edit.query_to_pandas_safe(query3)
deleted_comments_edit.head()