# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

oce_litigation = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="uspto_oce_litigation")
# View table names under the uspto_oce_litigation data table
bq_assistant = BigQueryHelper("patents-public-data", "uspto_oce_litigation")
bq_assistant.list_tables()
# View the first three rows of the cases data table
bq_assistant.head("cases", num_rows=3)
# View information on all columns in the cases data table
bq_assistant.table_schema("cases")
query1 = """
SELECT DISTINCT
  case_cause
FROM
  `patents-public-data.uspto_oce_litigation.cases`
LIMIT
  20;
        """
response1 = oce_litigation.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)