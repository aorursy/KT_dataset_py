import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
QUERY = """
    SELECT
      name, 
      count(name) AS name_count
    FROM
      `bigquery-public-data.usa_names.usa_1910_current`
    GROUP BY name
    ORDER BY 2 DESC
    LIMIT 10
        """

bq_assistant = BigQueryHelper("bigquery-public-data", "usa_names")

df = bq_assistant.query_to_pandas(QUERY)

df.head()