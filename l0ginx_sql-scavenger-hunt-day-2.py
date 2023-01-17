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

popular_stories = hacker_news.query_to_pandas_safe(query)
popular_stories.head()

query2 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

type_of_story = hacker_news.query_to_pandas_safe(query2)
type_of_story.head()
query3 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted is TRUE
        """

commend_deleted = hacker_news.query_to_pandas_safe(query3)
commend_deleted.head()
query3 = """SELECT type,  
    MIN(score) as MIN_Score , MAX(score) as MAX_Score , AVG(score) as AVG_Score 
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

type_of_story = hacker_news.query_to_pandas_safe(query3)
type_of_story.head()