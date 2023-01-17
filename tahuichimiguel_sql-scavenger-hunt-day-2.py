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
query1 = """SELECT type, COUNT(id) AS story_count
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY story_count
        """
results1 = hacker_news.query_to_pandas_safe(query1)
print('Number of Stories per Type in `full` Table')
print(results1)

query2 = """SELECT COUNT(id) AS total_deleted
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = True
        """
results2 = hacker_news.query_to_pandas_safe(query2)
print('\nTotal Number of Deleted Comments in `comments` Table')
print(results2)

query3 = """SELECT 
                type AS detail_type
                , COUNT(id) AS total_deleted
                , MIN(timestamp) earliest_deletion_date
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY type
            ORDER BY total_deleted 
        """
results3 = hacker_news.query_to_pandas_safe(query3)
print('\nDeleted File Count and Earliest Deletion Date For Each Type in `full` Table')
print(results3)