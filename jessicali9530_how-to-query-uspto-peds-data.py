# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

peds = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="uspto_peds")
# View table names under the uspto_peds data table
bq_assistant = BigQueryHelper("patents-public-data", "uspto_peds")
bq_assistant.list_tables()
# View the first three rows of the applications data table
bq_assistant.head("applications", num_rows=3)
# View information on all columns in the wdi_2016 data table
bq_assistant.table_schema("applications")
query1 = """
SELECT DISTINCT
  patentCaseMetadata.applicantFileReference
FROM
  `patents-public-data.uspto_peds.applications`
LIMIT
  20;
        """
response1 = peds.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)