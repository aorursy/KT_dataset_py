import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
international_census = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="census_bureau_international")
bq_assistant = BigQueryHelper("bigquery-public-data", "census_bureau_international")
bq_assistant.list_tables()
bq_assistant.head("mortality_life_expectancy", num_rows=3)
bq_assistant.table_schema("mortality_life_expectancy")
query1 = """SELECT
  age.country_name,
  age.life_expectancy,
  size.country_area
FROM (
  SELECT
    country_name,
    life_expectancy
  FROM
    `bigquery-public-data.census_bureau_international.mortality_life_expectancy`
  WHERE
    year = 2016) age
INNER JOIN (
  SELECT
    country_name,
    country_area
  FROM
    `bigquery-public-data.census_bureau_international.country_names_area` where country_area > 25000) size
ON
  age.country_name = size.country_name
ORDER BY
  2 DESC
/* Limit removed for Data Studio Visualization */
LIMIT
  10
        """
response1 = international_census.query_to_pandas_safe(query1)
response1.head(10)
query2 = """SELECT
  age.country_name,
  SUM(age.population) AS under_25,
  pop.midyear_population AS total,
  ROUND((SUM(age.population) / pop.midyear_population) * 100,2) AS pct_under_25
FROM (
  SELECT
    country_name,
    population,
    country_code
  FROM
    `bigquery-public-data.census_bureau_international.midyear_population_agespecific`
  WHERE
    year =2017
    AND age < 25) age
INNER JOIN (
  SELECT
    midyear_population,
    country_code
  FROM
    `bigquery-public-data.census_bureau_international.midyear_population`
  WHERE
    year = 2017) pop
ON
  age.country_code = pop.country_code
GROUP BY
  1,
  3
ORDER BY
  4 DESC
/* Remove limit for visualization */
LIMIT
  10
        """
response2 = international_census.query_to_pandas_safe(query2)
response2.head(10)
query3 = """SELECT
  growth.country_name,
  growth.net_migration,
  CAST(area.country_area as INT64) as country_area
FROM (
  SELECT
    country_name,
    net_migration,
    country_code
  FROM
    `bigquery-public-data.census_bureau_international.birth_death_growth_rates`
  WHERE
    year = 2017) growth
INNER JOIN (
  SELECT
    country_area,
    country_code
  FROM
    `bigquery-public-data.census_bureau_international.country_names_area`
  WHERE
    country_area > 500) area
ON
  growth.country_code = area.country_code
ORDER BY
  net_migration DESC
LIMIT
  10
        """
response3 = international_census.query_to_pandas_safe(query3, max_gb_scanned=10)
response3.head(10)
