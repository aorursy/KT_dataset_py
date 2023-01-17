print('What tables do I have?')

import pandas as pd

from bq_helper import BigQueryHelper

bq_assistant = BigQueryHelper("bigquery-public-data", "google_analytics_sample")

bq_assistant.list_tables()[:5]
QUERY = """

    SELECT 

        *  -- Warning, be careful when doing SELECT ALL

    FROM `bigquery-public-data.google_analytics_sample.ga_sessions_20160801` 

    LIMIT 10

"""



df = bq_assistant.query_to_pandas(QUERY)

df.head()