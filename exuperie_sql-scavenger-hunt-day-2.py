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
# select type, count based on group by type  
query_one = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
#run query safely
story_type = hacker_news.query_to_pandas_safe(query_one)
#display first few rows
story_type.head()

#select deleted column,count based on group by 'deleted'value
#only return counts of TRUE
query_two = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted
            HAVING deleted = TRUE
        """
deleted_comments = hacker_news.query_to_pandas_safe(query_two)
deleted_comments.head()

#which story type has the most amount of stories?
query_three = """SELECT type, MAX(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type 
        """
#run query safely
most_type = hacker_news.query_to_pandas_safe(query_three)
#display first few rows
most_type.head()
