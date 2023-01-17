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
# how many stories of each type
query1 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
n_stories = hacker_news.query_to_pandas_safe(query1)
print(n_stories)
# how many comments have been deleted
query2 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = TRUE
        """
n_deleted = hacker_news.query_to_pandas_safe(query2)
print(n_deleted)
# what is the best score for stories by each author where authors have more than 20 stories
query3 = """ SELECT author, MAX(score)
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY author
            HAVING COUNT(id) > 20
          """
scores = hacker_news.query_to_pandas_safe(query3)
print(scores)