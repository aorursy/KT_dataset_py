import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                       dataset_name='hacker_news')

query_1 = """SELECT type, COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type
          """

num_by_type = hacker_news.query_to_pandas_safe(query_1)
num_by_type
query_2 = """SELECT deleted, COUNT(id)
             FROM `bigquery-public-data.hacker_news.full`
             WHERE deleted is TRUE
             GROUP BY deleted
          """

num_deleted = hacker_news.query_to_pandas_safe(query_2)
num_deleted