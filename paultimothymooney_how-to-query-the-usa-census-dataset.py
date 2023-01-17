import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
census_data = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="census_bureau_usa")
bq_assistant = BigQueryHelper("bigquery-public-data", "census_bureau_usa")
bq_assistant.list_tables()
bq_assistant.head("population_by_zip_2010", num_rows=3)
bq_assistant.table_schema("population_by_zip_2010")
query1 = """SELECT
  zipcode,
  population
FROM
  `bigquery-public-data.census_bureau_usa.population_by_zip_2010`
WHERE
  gender = ''
ORDER BY
  population DESC
LIMIT
  10
        """
response1 = census_data.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT
  zipcode,
  pop_2000,
  pop_2010,
  pop_chg,
  pop_pct_chg
FROM (
  SELECT
    r1.zipcode AS zipcode,
    r2.population AS pop_2000,
    r1.population AS pop_2010,
    r1.population - r2.population AS pop_chg,
    ROUND((r1.population - r2.population)/NULLIF(r2.population,0) * 100, 2) AS pop_pct_chg,
    ABS((r1.population - r2.population)/NULLIF(r2.population,0)) AS abs_pct_chg
  FROM
    `bigquery-public-data.census_bureau_usa.population_by_zip_2010` AS r1
  INNER JOIN
    `bigquery-public-data.census_bureau_usa.population_by_zip_2000` AS r2
  ON
    r1.zipcode = r2.zipcode WHERE --following criteria selects total population without breaking down by age/gender
    r1.minimum_age IS NULL
    AND r2.minimum_age IS NULL
    AND r1.maximum_age IS NULL
    AND r2.maximum_age IS NULL
    AND r1.gender = ''
    AND r2.gender = '' )
ORDER BY
  abs_pct_chg DESC
LIMIT
  10
        """
response2 = census_data.query_to_pandas_safe(query2)
response2.head(10)