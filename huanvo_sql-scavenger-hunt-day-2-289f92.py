# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments", num_rows=10)
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
# print the first couple rows of the "full" table
hacker_news.head("full")
#query to find the number of stories in each type
story_query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
story_types = hacker_news.query_to_pandas_safe(story_query)
#check the head of story_type
story_types.head()
#query to find the number of deleted comments 
deleted_query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY deleted
            HAVING deleted = True
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
deleted_comments = hacker_news.query_to_pandas_safe(deleted_query)
deleted_comments
#query to find the max ranking of each author in the "comments" table
max_query = """SELECT author, MAX(ranking)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
            HAVING MAX(ranking) != 0
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
max_ranking = hacker_news.query_to_pandas_safe(max_query)
max_ranking.head()