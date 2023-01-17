# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

oce_office_actions = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="uspto_oce_office_actions")
# View table names under the uspto_oce_office_actions data table
bq_assistant = BigQueryHelper("patents-public-data", "uspto_oce_office_actions")
bq_assistant.list_tables()
# View the first three rows of the trials data table
bq_assistant.head("rejections", num_rows=3)
# View information on all columns in the trials data table
bq_assistant.table_schema("rejections")
query1 = """
SELECT DISTINCT
  action_subtype
FROM
  `patents-public-data.uspto_oce_office_actions.rejections`
LIMIT
  20;
        """
response1 = oce_office_actions.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)