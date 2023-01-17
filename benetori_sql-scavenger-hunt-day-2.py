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
result_query = """SELECT type, COUNT(id)
                  FROM `bigquery-public-data.hacker_news.full`
                  GROUP BY type
               """

result = hacker_news.query_to_pandas_safe(result_query)

result.head()
result_query_2 = """SELECT COUNT(deleted)
                  FROM `bigquery-public-data.hacker_news.full`
                  WHERE deleted = TRUE
               """

result_2 = hacker_news.query_to_pandas_safe(result_query_2)

result_2.head()
result_query_3 = """SELECT FORMAT_DATETIME("%Y",CAST(timestamp as DATETIME)) as year, type ,
                    MAX(score) as max_score, MIN(score) as min_score, AVG(score) as avg_score
                    FROM `bigquery-public-data.hacker_news.full`
                    GROUP BY year, type
                    HAVING year >= '2016'
                    ORDER BY year DESC
                 """

result_3 = hacker_news.query_to_pandas_safe(result_query_3)

result_3.head(100)