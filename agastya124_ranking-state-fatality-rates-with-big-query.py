import pandas as pd

from google.cloud import bigquery

from bq_helper import BigQueryHelper
QUERY = """SELECT

  state_name,

  COUNT(consecutive_number) AS accidents,

  SUM(number_of_fatalities) AS fatalities,

  SUM(number_of_fatalities) / COUNT(consecutive_number) AS fatalities_per_accident

FROM

  `bigquery-public-data.nhtsa_traffic_fatalities.accident_{0}`

GROUP BY

  state_name

ORDER BY

  fatalities_per_accident DESC"""
bq_assistant = BigQueryHelper("bigquery-public-data", "nhtsa_traffic_fatalities")
bq_assistant.estimate_query_size(QUERY.format(2015))
df_2015 = bq_assistant.query_to_pandas(QUERY.format(2015))
df_2015.head()
df_2016 = bq_assistant.query_to_pandas(QUERY.format(2016))
df_2016.head()
type(df_2016)
df_2016.describe()
df_2016.shape
!ls