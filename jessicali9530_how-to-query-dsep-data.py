# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

dsep = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="dsep")
# View table names under the uspto_ptab data table
bq_assistant = BigQueryHelper("patents-public-data", "dsep")
bq_assistant.list_tables()
# View the first three rows of the disclosures_13 data table
bq_assistant.head("disclosures_13", num_rows=3)
# View information on all columns in the disclosures_13 data table
bq_assistant.table_schema("disclosures_13")
query1 = """
SELECT DISTINCT
  patent_owner_harmonized, patent_owner_unharmonized
FROM
  `patents-public-data.dsep.disclosures_13`
LIMIT
  20;
        """
response1 = dsep.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)