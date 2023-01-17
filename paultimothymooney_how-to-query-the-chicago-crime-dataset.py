import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
chicago_crime = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="chicago_crime")
bq_assistant = BigQueryHelper("bigquery-public-data", "chicago_crime")
bq_assistant.list_tables()
bq_assistant.head("crime", num_rows=3)
bq_assistant.table_schema("crime")
query1 = """SELECT
  primary_type,
  description,
  COUNTIF(year = 2015) AS arrests_2015,
  COUNTIF(year = 2016) AS arrests_2016,
  FORMAT('%3.2f', (COUNTIF(year = 2016) - COUNTIF(year = 2015)) / COUNTIF(year = 2015)*100) AS pct_change_2015_to_2016
FROM
  `bigquery-public-data.chicago_crime.crime`
WHERE
  arrest = TRUE
  AND year IN (2015,
    2016)
GROUP BY
  primary_type,
  description
HAVING
  COUNTIF(year = 2015) > 100
ORDER BY
  (COUNTIF(year = 2016) - COUNTIF(year = 2015)) / COUNTIF(year = 2015) DESC
        """
response1 = chicago_crime.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT
  year,
  month,
  incidents
FROM (
  SELECT
    year,
    EXTRACT(MONTH
    FROM
      date) AS month,
    COUNT(1) AS incidents,
    RANK() OVER (PARTITION BY year ORDER BY COUNT(1) DESC) AS ranking
  FROM
    `bigquery-public-data.chicago_crime.crime`
  WHERE
    primary_type = 'MOTOR VEHICLE THEFT'
    AND year <= 2016
  GROUP BY
    year,
    month )
WHERE
  ranking = 1
ORDER BY
  year DESC
        """
response2 = chicago_crime.query_to_pandas_safe(query2)
response2.head(10)