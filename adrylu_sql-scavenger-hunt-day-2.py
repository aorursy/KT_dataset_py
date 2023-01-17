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

query1=""" SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
type_stories=hacker_news.query_to_pandas_safe(query1)

print(type_stories)

query2=""" SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments` 
            GROUP BY deleted
            HAVING deleted=True            
            """
deleted_comments=hacker_news.query_to_pandas_safe(query2)

print(deleted_comments)
query3="""SELECT author, MAX(score), AVG(score), count(id)
            FROM `bigquery-public-data.hacker_news.stories`
            GROUP BY author
            HAVING count(id)>100
            ORDER BY AVG(score) desc 
        """

author_score=hacker_news.query_to_pandas_safe(query3)

print('Authors with the highest average score and more than 100 stories')
author_score.head()
            
    