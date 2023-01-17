import bq_helper

hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.list_tables()



hacker_news.head('stories')
query = """SELECT distinct type
            FROM `bigquery-public-data.hacker_news.stories`
        """

just_trying = hacker_news.query_to_pandas_safe(query)
just_trying.type.value_counts()
hacker_news.head('full')
query2 = """SELECT type, COUNT(id)
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
        """
count_types = hacker_news.query_to_pandas_safe(query2)
count_types.head()

hacker_news.head('comments')
query3 = """SELECT COUNT(id)
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted = TRUE
        """
deleted_comments = hacker_news.query_to_pandas_safe(query3)
deleted_comments.head()