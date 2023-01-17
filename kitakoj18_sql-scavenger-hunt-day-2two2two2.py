import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import bq_helper
hacker_news = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="hacker_news")
hacker_news.list_tables()
hacker_news.table_schema('full')
hacker_news.head('full')
types_query = """
              SELECT type, COUNT(id) AS story_count
                  FROM `bigquery-public-data.hacker_news.full`
              GROUP BY type
              """
types = hacker_news.query_to_pandas_safe(types_query)
types.head()
deleted_query = """
                SELECT COUNT(deleted) AS num_deleted_comments
                    FROM `bigquery-public-data.hacker_news.full`
                WHERE deleted = True
                """

deleted_com_count = hacker_news.query_to_pandas_safe(deleted_query)
deleted_com_count.head()
avg_score_query = """
                  SELECT `by` AS user, AVG(COALESCE(score, 0)) as avg_score
                      FROM `bigquery-public-data.hacker_news.full`
                  GROUP BY `by`
                  ORDER BY avg_score DESC
                  """

avg_scores = hacker_news.query_to_pandas_safe(avg_score_query)
avg_scores.head()
