# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# print the first couple rows of the "comments" table
hacker_news.head("full")
# import package with helper functions 
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print the first couple rows of the "comments" table
hacker_news.head("comments")

query = """SELECT type, COUNT(ID)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
stories_type = hacker_news.query_to_pandas_safe(query)
stories_type.head()

# import package with helper functions 
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print the first couple rows of the "comments" table
hacker_news.head("comments")

# must call group by so IDs are counted consecutive
query = """SELECT deleted, COUNT(ID)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY deleted 
        """

comments_deleted = hacker_news.query_to_pandas_safe(query)
print(comments_deleted)
# import package with helper functions 
import bq_helper
# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
# print the first couple rows of the "comments" table
hacker_news.head("comments")

# must call group by so IDs are counted consecutive
query = """SELECT deleted, MAX(ID)
            FROM `bigquery-public-data.hacker_news.full`
            WHERE deleted = True
            GROUP BY deleted 
        """

comments_deleted = hacker_news.query_to_pandas_safe(query)
print(comments_deleted)