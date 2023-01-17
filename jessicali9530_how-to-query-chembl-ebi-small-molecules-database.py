# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

ebi_chembl = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="ebi_chembl")
# View table names under the ebi_chembl data table
bq_assistant = BigQueryHelper("patents-public-data", "ebi_chembl")
bq_assistant.list_tables()
# View the first three rows of the tissue_dictionary_23 data table
bq_assistant.head("tissue_dictionary_23", num_rows=3)
# View information on all columns in the tissue_dictionary_23 data table
bq_assistant.table_schema("tissue_dictionary_23")
query1 = """
SELECT DISTINCT
  pref_name
FROM
  `patents-public-data.ebi_chembl.tissue_dictionary_23`
LIMIT
  20;
        """
response1 = ebi_chembl.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)