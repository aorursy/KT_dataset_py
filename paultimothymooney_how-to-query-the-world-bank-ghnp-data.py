import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
ghnp = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="world_bank_health_population")
bq_assistant = BigQueryHelper("bigquery-public-data", "world_bank_health_population")
bq_assistant.list_tables()
bq_assistant.head("health_nutrition_population", num_rows=30)
bq_assistant.table_schema("health_nutrition_population")
query1 = """
SELECT
  country_name,
  ROUND(AVG(value),2) AS average
FROM
  `bigquery-public-data.world_bank_health_population.health_nutrition_population`
WHERE
  indicator_code = "SP.DYN.SMAM.FE"
  AND year > 2000
GROUP BY
  country_name
ORDER BY
  average
;
        """
response1 = ghnp.query_to_pandas_safe(query1)
response1.head(50)