# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("comments")
hacker_news.head("full")
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
# query to pass to 
full_query = """SELECT type, COUNT(id) as Stories
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type            
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
full_stories = hacker_news.query_to_pandas_safe(full_query)
full_stories
# query to pass to 
cd_query = """SELECT deleted , COUNT(*) as deleted_count
            FROM `bigquery-public-data.hacker_news.comments`            
            GROUP BY deleted  
            HAVING deleted = True
        """
# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
cd_stories = hacker_news.query_to_pandas_safe(cd_query)
cd_stories