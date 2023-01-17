# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the list of tables in Hacker News dataset
hacker_news.list_tables()
hacker_news.head('full')
# query to pass to 
query_1 = """SELECT type, COUNT(id) as Counts
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
# query_to_pandas_safe method will cancel the query if it would use too much of our quota,
# with the limit set to 1 GB by default
type_counts = hacker_news.query_to_pandas_safe(query_1)
type_counts.head(10)
hacker_news.head('comments')
query_2 = """ SELECT COUNT(id) as TotalCount, 
              SUM(CASE WHEN deleted = True then 1 else 0 end) as DeletedCount
              FROM `bigquery-public-data.hacker_news.comments`
          """
hacker_news.estimate_query_size(query_2)
deleted_counts = hacker_news.query_to_pandas_safe(query_2)
deleted_counts.head()