import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package
irs = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="irs_990")
bq_assistant = BigQueryHelper("bigquery-public-data", "irs_990")
bq_assistant.list_tables()
bq_assistant.head("irs_990_2016", num_rows=10)
bq_assistant.table_schema("irs_990_2016")
query1 = """SELECT
  irsein.name AS name,
  irsein.state AS state,
  irsein.city AS city,
  irs990.totrevenue AS revenue,
  irs990.noemplyeesw3cnt AS employees,
  irs990.noindiv100kcnt AS employees_over_100k,
  irs990.compnsatncurrofcr AS officers_comp
FROM
  `bigquery-public-data.irs_990.irs_990_ein` AS irsein
JOIN
  `bigquery-public-data.irs_990.irs_990_2015` AS irs990
USING (ein)
ORDER BY
  revenue DESC;
        """
response1 = irs.query_to_pandas_safe(query1, max_gb_scanned=10)
response1.head(50)