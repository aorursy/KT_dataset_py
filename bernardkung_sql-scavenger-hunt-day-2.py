# import package with helper functions 
import bq_helper as bqh

# create a helper object for this dataset
hacker_news = bqh.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
# print the first couple rows of the "full" table
hacker_news.head("full")
# Your code goes here :)
query1 = """
    SELECT type, COUNT(id) AS count
    FROM `bigquery-public-data.hacker_news.full`
    GROUP BY type
    """
hacker_news.estimate_query_size(query1)
type_table = hacker_news.query_to_pandas_safe(query1)
print(type_table)
# print the first couple rows of the "comments" table
hacker_news.head("comments")
query2 = """
    SELECT COUNT(id) AS deleted_comments
    FROM `bigquery-public-data.hacker_news.comments`
    WHERE deleted = True
    """
hacker_news.estimate_query_size(query2)
hacker_news.query_to_pandas_safe(query2, max_gb_scanned = 0.2)
query3 = """
    SELECT `by`, COUNT(DISTINCT(title)) AS num_stories, AVG(score) AS avg_score
    FROM `bigquery-public-data.hacker_news.full`
    WHERE type = "story"
    GROUP BY `by`
    HAVING avg_score > 800 AND num_stories > 1
    ORDER BY num_stories DESC
    """
hacker_news.estimate_query_size(query3)
hacker_news.query_to_pandas_safe(query3, max_gb_scanned = 0.5)