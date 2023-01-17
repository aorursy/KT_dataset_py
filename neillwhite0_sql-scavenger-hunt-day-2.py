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
# First off, what does the "full" table look like?
# print the first couple rows of the "full" table
hacker_news.head("full")
# Q1: How many stories (use the "id" column) are there of each type (in the "type" column) 
#     in the full table?
# Question 1 Query 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
type_counts = hacker_news.query_to_pandas_safe(query)
print(type_counts)

import matplotlib.pyplot as plt
plt.bar(type_counts['type'],type_counts['f0_'],log=True)
# there are so many comments compared to pollopt type, job, and poll, we can't see them
# so try a log scale
plt.xlabel('Content Type')
plt.ylabel('Log(Count)')
plt.title('Hacker News Stories')
# Q2: How many comments have been deleted? (If a comment was deleted the "deleted" column in the 
#     comments table will have the value "True".)

# Question 2 Query 
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(query)
print(deleted_comments)

# Optional extra credit Q: read about [aggregate functions other than COUNT()] 
# and modify one of the queries you wrote above to use a different aggregate function.

# Hmmmm, how about, "What's the max number of descendants of each content type?"
# Extra Credit Query 
query = """SELECT type, MAX(descendants)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
max_descendants = hacker_news.query_to_pandas_safe(query)
print(max_descendants)

plt.bar(max_descendants['type'],max_descendants['f0_'])
plt.xlabel('Content Type')
plt.ylabel('Max descendants')
plt.title('Hacker News Stories')