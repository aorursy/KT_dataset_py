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
hacker_news.head("full")
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
story_type = hacker_news.query_to_pandas_safe(query)
story_type
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted 
        """
nb_deleted = hacker_news.query_to_pandas_safe(query)
nb_deleted
query = """ SELECT SUM(score) as total_score, author, count(id) as total_stories
            FROM `bigquery-public-data.hacker_news.stories`
            WHERE author IS NOT NULL
            GROUP BY author
            ORDER BY total_stories DESC
            LIMIT 30
        """

nb_story = hacker_news.query_to_pandas_safe(query)
nb_story