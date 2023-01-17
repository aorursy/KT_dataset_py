# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
# hacker_news.head("full")

Q1 = """SELECT type AS StoryType, COUNT(type) AS Amount
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            LIMIT 100
        """
type_of_stories = hacker_news.query_to_pandas_safe(Q1)
type_of_stories.head()

Q2 = """SELECT deleted, COUNT(id) AS Amount
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
            GROUP BY deleted 
            LIMIT 100
        """
deleted_comments = hacker_news.query_to_pandas_safe(Q2)
deleted_comments.head()

Q3 = """SELECT author, COUNT(author) AS NumberOfArticles
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            LIMIT 100
        """
story_type_sum = hacker_news.query_to_pandas_safe(Q3)
story_type_sum.head()
