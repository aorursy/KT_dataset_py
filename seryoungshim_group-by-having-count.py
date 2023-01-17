import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                      dataset_name="hacker_news")
hacker_news.head("comments")
query="""SELECT parent, COUNT(id)
         FROM `bigquery-public-data.hacker_news.comments`
         GROUP BY parent
         HAVING COUNT(id) < 10
         """
popular_stories=hacker_news.query_to_pandas_safe(query)
print(popular_stories.head())