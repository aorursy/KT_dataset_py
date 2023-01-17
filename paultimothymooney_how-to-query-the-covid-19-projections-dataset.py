import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

sampleTables = bq_helper.BigQueryHelper(active_project="bigquery-public-data",

                                   dataset_name="covid19_public_forecasts")
bq_assistant = BigQueryHelper("bigquery-public-data", "covid19_public_forecasts")

bq_assistant.list_tables()
bq_assistant.head("county_14d", num_rows=20)
query1 = """SELECT

  *

FROM

  `bigquery-public-data.covid19_public_forecasts.state_14d` 

WHERE

  state_fips_code = "48"

  AND prediction_date >= forecast_date

ORDER BY

  prediction_date

        """

response1 = sampleTables.query_to_pandas_safe(query1, max_gb_scanned=10)

response1.head(10)
query2 = """SELECT

  *

FROM

  `bigquery-public-data.covid19_public_forecasts.county_14d` 

WHERE

  state_name = "New York"

  AND prediction_date > forecast_date

ORDER BY

  prediction_date

        """

response2 = sampleTables.query_to_pandas_safe(query2, max_gb_scanned=10)

response2.head(20)