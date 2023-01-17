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
# Your code goes here :)
# stories of each type in "full" table
type_query = """SELECT type, COUNT(id) as type_count
                FROM `bigquery-public-data.hacker_news.full`
                GROUP BY type
             """
type_count = hacker_news.query_to_pandas_safe(type_query)
type_count.head()

# deleted comments count
deleted_query = """SELECT deleted, COUNT(id) as deleted_count
                    FROM `bigquery-public-data.hacker_news.comments`
                    WHERE deleted = True
                    GROUP BY deleted
                """
deleted_count = hacker_news.query_to_pandas_safe(deleted_query)
deleted_count.head()
# use of another aggregate function
# max function
max_query = """SELECT MAX(ranking) AS max_ranking
                FROM `bigquery-public-data.hacker_news.comments`
            """
max_ranking = hacker_news.query_to_pandas_safe(max_query)
max_ranking.head()
# sum function
hacker_news.head("comments")
sum_query = """SELECT id, SUM(ranking) as sum_ranking
                FROM `bigquery-public-data.hacker_news.comments`
                GROUP BY id"""
sum_ranking = hacker_news.query_to_pandas_safe(sum_query)
sum_ranking.head()