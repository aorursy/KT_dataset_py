# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

marec = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="marec")
# View table names under the marec data table
bq_assistant = BigQueryHelper("patents-public-data", "marec")
bq_assistant.list_tables()
# View the first three rows of the publications data table
bq_assistant.head("publications", num_rows=3)
# View information on all columns in the publications data table
bq_assistant.table_schema("publications")
query1 = """
SELECT DISTINCT
  publication_number
FROM
  `patents-public-data.marec.publications`
LIMIT
  20;
        """
response1 = marec.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)