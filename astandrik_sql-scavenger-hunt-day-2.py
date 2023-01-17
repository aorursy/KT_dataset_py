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
hacker_news.head('full').columns
# How many stories (use the "id" column) are there of each type (in the "type" column) in the full table?
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

total_stories_by_type = hacker_news.query_to_pandas_safe(query1)
total_stories_by_type
# How many comments have been deleted? 
query2 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is not null
        """

total_comments_deleted = hacker_news.query_to_pandas_safe(query2)
total_comments_deleted
# **Optional extra credit**: Modified aggregation using Average.
#Average score by type
query3 = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE score is not null
            GROUP BY type
        """

avg_score_by_type = hacker_news.query_to_pandas_safe(query3)
avg_score_by_type
