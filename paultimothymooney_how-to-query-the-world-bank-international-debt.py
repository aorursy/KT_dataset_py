import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
wbid = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="world_bank_intl_debt")
bq_assistant = BigQueryHelper("bigquery-public-data", "world_bank_intl_debt")
bq_assistant.list_tables()
bq_assistant.head("international_debt", num_rows=15)
bq_assistant.table_schema("international_debt")
query1 = """
SELECT
  id.country_name,
  id.value AS debt --format in DataStudio
FROM (
  SELECT
    country_code,
    region
  FROM
    `bigquery-public-data.world_bank_intl_debt.country_summary`
  WHERE
    region != "" ) cs --aggregated countries do not have a region
INNER JOIN (
  SELECT
    country_code,
    country_name,
    value,
    year
  FROM
    `bigquery-public-data.world_bank_intl_debt.international_debt`
  WHERE
    indicator_code = "DT.DOD.PVLX.CD"
    AND year = 2016 ) id
ON
  cs.country_code = id.country_code
ORDER BY
  id.value DESC
;
        """
response1 = wbid.query_to_pandas_safe(query1)
response1.head(50)