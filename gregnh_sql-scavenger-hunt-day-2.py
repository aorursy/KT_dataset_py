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
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type"""
hacker_news.estimate_query_size(query)
count_stories_per_type = hacker_news.query_to_pandas_safe(query)
count_stories_per_type.to_csv('count_stories_per_type.csv')
count_stories_per_type.head()
query = """SELECT deleted, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY deleted"""
hacker_news.estimate_query_size(query)
total_deleted = hacker_news.query_to_pandas_safe(query)
total_deleted.to_csv('total_deleted.csv')
total_deleted
query = """SELECT type, AVG(score), STDDEV(score)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type"""
hacker_news.estimate_query_size(query)
stat_type = hacker_news.query_to_pandas_safe(query)
print(stat_type)