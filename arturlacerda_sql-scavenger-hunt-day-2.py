import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                       dataset_name="hacker_news")
hacker_news.head("comments")
hacker_news.head("full")
query = """SELECT type, COUNT(id) AS num_stories
           FROM `bigquery-public-data.hacker_news.full`
           GROUP BY type
        """
stories_by_type = hacker_news.query_to_pandas_safe(query)
stories_by_type
query = """SELECT COUNT(id)
           FROM `bigquery-public-data.hacker_news.comments`
           WHERE deleted = True
        """
hacker_news.query_to_pandas_safe(query)