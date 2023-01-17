import bq_helper as bq
hacker_news = bq.BigQueryHelper(active_project= "bigquery-public-data", dataset_name= "hacker_news" )
hacker_news.list_tables()
hacker_news.head("full")
query1 = """ SELECT type, count(id)
             FROM `bigquery-public-data.hacker_news.full`
             GROUP BY type
         """
q1_output= hacker_news.query_to_pandas_safe(query1)
q1_output.head()
hacker_news.head("comments")
query2 = """ SELECT count(id)
             FROM `bigquery-public-data.hacker_news.comments`
             WHERE deleted=true
         """
q2_output= hacker_news.query_to_pandas_safe(query2)
q2_output