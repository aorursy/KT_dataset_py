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
# Your code goes here :)
hacker_news.head("full")
# query for finding counts of story types 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_types = hacker_news.query_to_pandas_safe(query)
story_types.head()
# query for finding how many comments were deleted
query = """SELECT deleted, 
                COUNT(id) AS `Count`, 
                (COUNT(id) * 100 / (SELECT COUNT(*) FROM `bigquery-public-data.hacker_news.comments`)) AS `Percentage`
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
deleted_comments = hacker_news.query_to_pandas_safe(query)
deleted_comments.head()
# query for finding user with largest "total score for stories"
query = """SELECT `by` as `User`, 
                SUM(score) AS `Total_Score`
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY User
            ORDER BY Total_Score DESC
        """
highest_scores = hacker_news.query_to_pandas_safe(query)
highest_scores.head()