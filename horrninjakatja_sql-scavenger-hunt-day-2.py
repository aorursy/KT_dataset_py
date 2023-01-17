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
# print the first couple rows of the "comments" table
hacker_news.head("comments")

# stories of each author (type column is missing?!)
authorquery = """SELECT author, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
        """
stories_per_author = hacker_news.query_to_pandas_safe(authorquery)
stories_per_author

# author's highest ranking (optional)
maxauthorquery = """SELECT author,MAX(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
        """ 
maxranking_per_author = hacker_news.query_to_pandas_safe(maxauthorquery)
maxranking_per_author

# deleted comments
delquery = """SELECT deleted,count(id)
            FROM  `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted=TRUE
        """
deleted_columns = hacker_news.query_to_pandas_safe(delquery)
deleted_columns


