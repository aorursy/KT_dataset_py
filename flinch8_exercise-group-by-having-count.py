# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
query ="""
        SELECT type, COUNT(id)
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """

type_count = hacker_news.query_to_pandas_safe(query)
print(type_count.head())
query2 ="""
        SELECT COUNT(id)
        FROM `bigquery-public-data.hacker_news.comments`
        WHERE deleted is True
        """

deleted_count = hacker_news.query_to_pandas_safe(query2)
print(deleted_count.head())
#Displaying the max score for each type 
query3 ="""
        SELECT type, MAX(score) as MAX_Score
        FROM `bigquery-public-data.hacker_news.full`
        GROUP BY type
        """

max_score = hacker_news.query_to_pandas_safe(query3)
print(max_score.head())