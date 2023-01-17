# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

investigations = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="usitc_investigations")
# View table names under the usitc_investigations
bq_assistant = BigQueryHelper("patents-public-data", "usitc_investigations")
bq_assistant.list_tables()
# View the first three rows of the investigations data table
bq_assistant.head("investigations", num_rows=3)
# View information on all columns in the investigations data table
bq_assistant.table_schema("investigations")
query1 = """
SELECT DISTINCT
  currentStatus
FROM
  `patents-public-data.usitc_investigations.investigations`
LIMIT
  20;
        """
response1 = investigations.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)