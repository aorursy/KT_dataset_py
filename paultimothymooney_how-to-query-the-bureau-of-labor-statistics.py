import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
BLS = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="bls")
bq_assistant = BigQueryHelper("bigquery-public-data", "bls")
bq_assistant.list_tables()
bq_assistant.head('cpi_u', num_rows=3)
bq_assistant.table_schema("cpi_u")
query1 = """SELECT *, ROUND((100*(value-prev_year)/value), 1) rate
FROM (
  SELECT
    year,
    LAG(value) OVER(ORDER BY year) prev_year,
    ROUND(value, 1) AS value,
    area_name
  FROM
    `bigquery-public-data.bls.cpi_u`
  WHERE
    period = "S03"
    AND item_code = "SA0"
    AND area_name = "U.S. city average"
)
ORDER BY year
        """
response1 = BLS.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT
  year,
  date,
  period,
  value,
  series_title
FROM
  `bigquery-public-data.bls.unemployment_cps`
WHERE
  series_id = "LNS14000000"
  AND year = 2016
ORDER BY date
        """
response2 = BLS.query_to_pandas_safe(query2)
response2.head(10)
query3 = """SELECT
  year,
  period,
  value,
  series_title
FROM
  `bigquery-public-data.bls.wm`
WHERE
  series_title LIKE '%Pittsburgh, PA%'
  AND year = 2016
ORDER BY
  value DESC
LIMIT
  10
        """
response3 = BLS.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
