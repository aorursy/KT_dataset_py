# import package with helper functions 
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print the first five rows of the "full" table
hacker_news.head("full")
# * How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query1 = """SELECT type, COUNT(id) as ct_id, COUNT(distinct id) as ct_distinct_id
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY type
        """

# Estimate query size
hacker_news.estimate_query_size(query1)
# cancel the query if it exceeds 1 GB
type_stories = hacker_news.query_to_pandas_safe(query1)
print("number of stories by type")
print(type_stories)
# print the first five rows of the "comments" table
hacker_news.head("comments")
# How many comments have been deleted? 
# (If a comment was deleted, the "deleted" column in the comments table will have the value "True".)
query2 = """SELECT deleted, sum(1) as count_comment
            FROM `bigquery-public-data.hacker_news.comments`
            group by 1
        """
# Estimate query size
hacker_news.estimate_query_size(query2)
# cancel the query if it exceeds 1 GB
del_comments = hacker_news.query_to_pandas_safe(query2)
print(del_comments)