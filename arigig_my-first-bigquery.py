# import our bq_helper package
import bq_helper
# create a helper object for our bigquery dataset
bitcoin_blocks = bq_helper.BigQueryHelper(active_project= "bigquery-public-data",
                                         dataset_name="bitcoin_blockchain")
# print a list of all the tables in the bitcoin_blocks dataset
bitcoin_blocks.list_tables()
# print information on all the columns in the "transactions" table
# in the hacker_news dataset
bitcoin_blocks.table_schema("transactions")
# preview the first couple lines of the "transactions" table
bitcoin_blocks.head("transactions")
# preview the first ten entries in the work_terahash column of the transactions table
bitcoin_blocks.head("transactions", selected_columns="work_terahash", num_rows=10)