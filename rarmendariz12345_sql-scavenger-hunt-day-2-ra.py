# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
stories_query = """ SELECT type, COUNT(id)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY(type)
                """
stories_type = hacker_news.query_to_pandas_safe(stories_query)
stories_type
deleted_query = """ SELECT type, COUNT(id)
                    FROM `bigquery-public-data.hacker_news.full`
                    WHERE type = 'comment' AND deleted = TRUE
                    GROUP BY(type)
                    """
deleted_comments = hacker_news.query_to_pandas_safe(deleted_query)
deleted_comments
deleted_query2 = """ SELECT COUNTIF(deleted)
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY(id)
                    """
deleted_comments_2 = hacker_news.query_to_pandas_safe(deleted_query)
deleted_comments_2