# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

cpc = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="cpc")
# View table names under the cpc data table
bq_assistant = BigQueryHelper("patents-public-data", "cpc")
bq_assistant.list_tables()
# View the first three rows of the definitions data table
bq_assistant.head("definitions", num_rows=3)
# View information on all columns in the definitions data table
bq_assistant.table_schema("definitions")
query1 = """
SELECT DISTINCT
  level
FROM
  `patents-public-data.cpc.definitions`
LIMIT
  20;
        """
response1 = cpc.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1) 