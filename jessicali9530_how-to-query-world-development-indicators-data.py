# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.
import bq_helper
from bq_helper import BigQueryHelper
# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package

wdi = bq_helper.BigQueryHelper(active_project="patents-public-data",
                                   dataset_name="worldbank_wdi")
# View table names under the worldbank_wdi data table
bq_assistant = BigQueryHelper("patents-public-data", "worldbank_wdi")
bq_assistant.list_tables()
# View the first three rows of the wdi_2016 data table
bq_assistant.head("wdi_2016", num_rows=3)
# View information on all columns in the wdi_2016 data table
bq_assistant.table_schema("wdi_2016")
query1 = """
SELECT
  country_name, indicator_value
FROM
  `patents-public-data.worldbank_wdi.wdi_2016`
ORDER BY 
  indicator_value DESC
LIMIT
  20;
        """
response1 = wdi.query_to_pandas_safe(query1)
response1.head(20)
bq_assistant.estimate_query_size(query1)