import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

hacker_news.list_tables()

hacker_news.head("full")

storytypes= """select type, count(id)
                FROM  `bigquery-public-data.hacker_news.full`
                GROUP BY type"""

Stypes = hacker_news.query_to_pandas_safe(storytypes)

Stypes.head()

hacker_news.head("comments")

delt= """select count(id)
         FROM  `bigquery-public-data.hacker_news.comments`
        where deleted = true"""
        
delete= hacker_news.query_to_pandas_safe(delt)

delete.head()


storytypes2= """select title, avg(score)
                FROM  `bigquery-public-data.hacker_news.full`
                GROUP BY title"""

Stypes2 = hacker_news.query_to_pandas_safe(storytypes2)

Stypes2.head()

hacker_news.head("comments")
