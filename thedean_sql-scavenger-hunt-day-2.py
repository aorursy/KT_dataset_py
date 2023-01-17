# Your code goes here :)

# import the package with helper functions
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name = "hacker_news")

# print the first handful of rows of the full table
hacker_news.head("full")

# generate our first query ("how many stories of each type")
type_query = """
            SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING count(id) > 0
            """
# generate a safe execution method

type_query = hacker_news.query_to_pandas_safe(type_query)
print(type_query.head())

# generate our second query (Count deleted comments)
deleted_query ="""SELECT
            COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = TRUE
            """

# generate a safe execution method

deleted_count = hacker_news.query_to_pandas_safe(deleted_query)
print(deleted_count.head())

# Extra credit, use another aggregate function. I want to try HAVING and ORDER BY too.

avg_score = """
            SELECT type, round(AVG(score))
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            HAVING round(AVG(score)) > 0
            ORDER BY round(AVG(score)) DESC
            """
# generate a safe execution method

avg_score_query = hacker_news.query_to_pandas_safe(avg_score)
print(avg_score_query)

