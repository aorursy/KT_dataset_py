# Start by importing the bq_helper module and calling on the specific active_project and dataset_name for the BigQuery dataset.

import bq_helper

from bq_helper import BigQueryHelper

# https://www.kaggle.com/sohier/introduction-to-the-bq-helper-package



patents = bq_helper.BigQueryHelper(active_project="patents-public-data",

                                   dataset_name="patents")
# View table names under the patents data table

bq_assistant = BigQueryHelper("patents-public-data", "patents")

bq_assistant.list_tables()
# View the first three rows of the publications data table

patent_df = bq_assistant.head("publications", num_rows=3)

patent_df.to_csv("patents.csv")
# View information on all columns in the publications data table

publications_data = bq_assistant.table_schema("publications")

publications_data.to_csv("publications_schema.csv")