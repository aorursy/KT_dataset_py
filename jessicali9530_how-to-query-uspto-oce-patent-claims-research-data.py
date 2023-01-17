# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

oce_claims = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="uspto_oce_claims")
# View table names under the uspto_oce_claims data table
bq_assistant = BigQueryHelper("patents-public-data", "uspto_oce_claims")
bq_assistant.list_tables()
# View the first three rows of the patent_claims_fulltext data table
bq_assistant.head("patent_claims_stats", num_rows=3)
# View information on all columns in the patent_claims_fulltext data table
bq_assistant.table_schema("patent_claims_stats")
query1 = """
SELECT
  AVG(CAST(word_ct AS INT64))
FROM
  `patents-public-data.uspto_oce_claims.patent_claims_stats`;
        """
response1 = oce_claims.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)