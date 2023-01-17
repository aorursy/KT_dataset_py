# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.table_schema("comments")
hacker_news.list_tables()
hacker_news.head("full")
# Your Code Here
query = """
        SELECT DISTINCT type,COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """
story_type_count = hacker_news.query_to_pandas(query)
print(story_type_count)
# Your Code Here
query = """SELECT COUNT(deleted)
           FROM `bigquery-public-data.hacker_news.comments`
           WHERE deleted = True
        """

deleted_comments = hacker_news.query_to_pandas(query)
print(deleted_comments)
# Your Code Here
query = """ SELECT COUNTIF(deleted=True)
            FROM `bigquery-public-data.hacker_news.comments`
        """
deleted_comments2 = hacker_news.query_to_pandas(query)
print(deleted_comments2)