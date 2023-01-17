import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")

hacker_news.head("comments")
query = """SELECT parent, COUNT(id) AS pop_stories
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY parent
            HAVING COUNT(id) > 10
        """

popular_stories = hacker_news.query_to_pandas_safe(query)

popular_stories.head()
query = """SELECT type, COUNT(id) AS num_pop_types
            FROM `bigquery-public-data.hacker_news.full`
            GROUP BY type
            ORDER BY COUNT(id) DESC
        """

popular_types = hacker_news.query_to_pandas_safe(query)

popular_types.head()
query = """SELECT COUNT(id) AS num_deleted_comments
            FROM `bigquery-public-data.hacker_news.comments`
            WHERE deleted IS True
        """

deleted = hacker_news.query_to_pandas_safe(query)

deleted
query = """SELECT author, MIN(time_ts) AS earliest_post, MAX(time_ts) AS latest_post
            FROM `bigquery-public-data.hacker_news.comments`
            GROUP BY author
        """

max_read_time = hacker_news.query_to_pandas_safe(query)

max_read_time.head()
