# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories = hacker_news.query_to_pandas_safe(query)
print(stories.head())
query = """SELECT COUNT(id) deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted
        """
deleted = hacker_news.query_to_pandas_safe(query)
print(deleted)
query = """SELECT type, avg(score) avg_score
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
average_story = hacker_news.query_to_pandas_safe(query)
print(average_story.head)