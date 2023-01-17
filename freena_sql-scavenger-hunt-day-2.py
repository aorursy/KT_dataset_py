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
# Number of stories of each type
query1 = """SELECT type, COUNT(id) AS Num
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_stories = hacker_news.query_to_pandas_safe(query1)
type_stories
# How many comments have been deleted ?
query2 = """SELECT COUNT(id) AS Num
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted is TRUE 
        """
deleted_stories = hacker_news.query_to_pandas_safe(query2)
deleted_stories
# Other Aggregate function Practice
# Which author spent the most time online?
query3 = """SELECT author, SUM(time) AS average_time
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY author
            ORDER BY average_time DESC
            LIMIT 10
         """
top_users = hacker_news.query_to_pandas_safe(query3)
top_users
# WHO got the highest stories score? 
query4 = """SELECT author, MAX(score) AS highest_score
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY author
            ORDER BY highest_score DESC
            LIMIT 10
         """
top_score = hacker_news.query_to_pandas_safe(query4)
top_score