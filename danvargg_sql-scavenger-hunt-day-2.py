import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                   dataset_name = "hacker_news")

hacker_news.head("full")
query = """SELECT parent, COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """
hacker_news.query_to_pandas_safe(query).head()
q2 = """SELECT type, COUNT(id) AS by_type
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY by_type DESC
        """
hacker_news.query_to_pandas_safe(q2).head()
q3 = """
    SELECT COUNT(id) AS deleted_comments
    FROM `bigquery-public-data.hacker_news.comments`
    WHERE deleted = True
"""
hacker_news.query_to_pandas_safe(q3)