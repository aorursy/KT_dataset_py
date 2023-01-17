# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

# query to pass to 
query = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """

# the query_to_pandas_safe method will cancel the query if
# it would use too much of your quota, with the limit set 
# to 1 GB by default
type_stories = hacker_news.query_to_pandas_safe(query)

type_stories
# import package with helper functions 
import bq_helper

# create a helper object for this dataset
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

query1 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = TRUE
        """

comments_deleted = hacker_news.query_to_pandas_safe(query1)

comments_deleted