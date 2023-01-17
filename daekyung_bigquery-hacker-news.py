import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

hacker_news.head("stories")
query = """
        SELECT *
        FROM `bigquery-public-data.hacker_news.stories`
        WHERE title != "" 
        ORDER BY time
        Limit 1000
        """

data = hacker_news.query_to_pandas_safe(query)

print(data.shape)
data.head()
data.to_pickle("./data.pkl")
