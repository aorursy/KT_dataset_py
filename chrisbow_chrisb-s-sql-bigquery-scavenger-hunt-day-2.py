# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# let's have a look at the first few rows of the comments table
hacker_news.head("comments")
# build query to count the number of ids from the full table
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# submit query using the safe, scan-size limited function
howManyStories = hacker_news.query_to_pandas_safe(query)

# print the result
howManyStories

# build query to count the number of ids from the comments table
query2 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted=True
        """
# submit query using the safe, scan-size limited function
deletedComments = hacker_news.query_to_pandas_safe(query2)

# print the result
deletedComments
hacker_news.head("full")
# build query to calculate the average score of types of article in the full table
query3 = """SELECT type, AVG(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# submit query using the safe, scan-size limited function
typeAverage = hacker_news.query_to_pandas_safe(query3)

# print the result
typeAverage
# build query to calculate the top score for each of article in the full table
query4 = """SELECT type, MAX(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# submit query using the safe, scan-size limited function
typeMax = hacker_news.query_to_pandas_safe(query4)

# print the result
typeMax
# build query to calculate the number of articles by type with scores over 100
query5 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE score > 100
            GROUP BY type
        """
# submit query using the safe, scan-size limited function
moreThanC = hacker_news.query_to_pandas_safe(query5)

# print the result
moreThanC