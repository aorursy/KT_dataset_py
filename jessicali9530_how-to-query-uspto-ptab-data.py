# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

ptab = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="uspto_ptab")
# View table names under the uspto_ptab data table
bq_assistant = BigQueryHelper("patents-public-data", "uspto_ptab")
bq_assistant.list_tables()
# View the first three rows of the trials data table
bq_assistant.head("trials", num_rows=3)
# View information on all columns in the trials data table
bq_assistant.table_schema("trials")
query1 = """
SELECT DISTINCT
  InventorName, ProsecutionStatus
FROM
  `patents-public-data.uspto_ptab.trials`
LIMIT
  20;
        """
response1 = ptab.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)