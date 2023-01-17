# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

patentsview = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="patentsview")
# View table names under the patentsview data table
bq_assistant = BigQueryHelper("patents-public-data", "patentsview")
bq_assistant.list_tables()
# View the first three rows of the patent data table
bq_assistant.head("patent", num_rows=3)
# View information on all columns in the patent data table
bq_assistant.table_schema("patent")
query1 = """
SELECT DISTINCT
  type
FROM
  `patents-public-data.patentsview.patent`
LIMIT
  20;
        """
response1 = patentsview.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)