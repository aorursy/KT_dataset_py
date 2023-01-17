import pandas as pd
import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data" ,
                                      dataset_name = "hacker_news")

# Listing tables
print(hacker_news.list_tables())
# Table schema for comments table
print("Comments Table:\n")
print(hacker_news.table_schema("comments"))

# Schema for the full table, which probably contains all the data
print("\nFull table:\n")
print(hacker_news.table_schema("full"))
# Schema for full_201510 table, which is probably a subset of full table for Oct 2015
print("\nOct 2015 Table:\n")
print(hacker_news.table_schema("full_201510"))

# Printing schema for stories table
print("\nStories Table:\n")
print(hacker_news.table_schema("stories"))
query = """
SELECT 
    COUNT(DISTINCT id)
FROM `bigquery-public-data.hacker_news.full`
"""

unique_stories = hacker_news.query_to_pandas_safe(query)
print("There are %s unique stories in the full table"%str(unique_stories.iloc[0][0]))

query11 = """
SELECT 
    type,COUNT(DISTINCT id) AS frequency
FROM `bigquery-public-data.hacker_news.full`
GROUP BY type
"""

unique_stories1 = hacker_news.query_to_pandas_safe(query11)
unique_stories1
query2 = """
SELECT 
    COUNT(id)
FROM `bigquery-public-data.hacker_news.comments`
WHERE deleted=True
"""

deleted_comments = hacker_news.query_to_pandas_safe(query2)
print("There are %s comments which were deleted according to the comments table"%str(deleted_comments.iloc[0][0]))
# BigQuery has another function called COUNTIF which counts within a particular group. The result of the 
# conditional statement must be a boolean. Since the column were are going to use already is boolean
# this is a good case to use it. So if I did HAVING DELETED = True, it gave an error. In standard SQL
# you'd have to write HAVING COUNTIF(deleted) = True however, countif is evaulating a condition, so doesnt
# make sense.

# In the documentation it says we can evaluate a condition in the countif expression, however when I tried
# COUNTIF(deleted=True), my RAM usage increased A LOT, so I killed the code

# It is interesting to note that it does not return False count

query3 = """
SELECT 
    COUNTIF(deleted)
FROM `bigquery-public-data.hacker_news.comments`
"""
deleted_comments2 = hacker_news.query_to_pandas_safe(query3)
print("There are %s comments which were deleted according to the comments table"%str(deleted_comments2.iloc[0][0]))

print("The statement deleted comments from COUNT query is same as the number from COUNTIF query is:")
print(deleted_comments.iloc[0][0]==deleted_comments2.iloc[0][0])