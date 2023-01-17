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
  *
FROM
  `bigquery-public-data.census_bureau_usa.population_by_zip_2010`

        """
response1 = census_data.query_to_pandas_safe(query1)
response1.to_csv("population_by_zip_2010.csv")
response1.head(10)

query2 = """SELECT
  *
FROM
  `bigquery-public-data.census_bureau_usa.population_by_zip_2000`

        """
response1 = census_data.query_to_pandas_safe(query2)
response1.to_csv("population_by_zip_2000.csv")
response1.head(10)

