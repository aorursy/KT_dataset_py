import pandas as pd
from google.cloud import bigquery
from bq_helper import BigQueryHelper
QUERY = """
    SELECT
        extract(DAYOFYEAR from date_local) as day_of_year,
        aqi
    FROM
      `bigquery-public-data.epa_historical_air_quality.pm25_frm_daily_summary`
    WHERE
      city_name = "Los Angeles"
      AND state_name = "California"
      AND sample_duration = "24 HOUR"
      AND poc = 1
      AND EXTRACT(YEAR FROM date_local) = 2015
    ORDER BY day_of_year
        """
bq_assistant = BigQueryHelper("bigquery-public-data", "epa_historical_air_quality")
df = bq_assistant.query_to_pandas(QUERY)
df.plot(x='day_of_year', y='aqi', style='.');
