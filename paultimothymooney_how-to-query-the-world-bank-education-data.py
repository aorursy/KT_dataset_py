import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
wbed = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="world_bank_intl_education")
bq_assistant = BigQueryHelper("bigquery-public-data", "world_bank_intl_education")
bq_assistant.list_tables()
bq_assistant.head("international_education", num_rows=30)
bq_assistant.table_schema("international_education")
query1 = """
SELECT
  country_name,
  AVG(value) AS average
FROM
  `bigquery-public-data.world_bank_intl_education.international_education`
WHERE
  indicator_code = "SE.XPD.TOTL.GB.ZS"
  AND year > 2000
GROUP BY
  country_name
ORDER BY
  average DESC
;
        """
response1 = wbed.query_to_pandas_safe(query1)
response1.head(50)