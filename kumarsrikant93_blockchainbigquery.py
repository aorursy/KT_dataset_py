# import our bq_helper package
import bq_helper 
# create a helper object for our bigquery dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "bitcoin_blockchain")
# print a list of all the tables in the hacker_news dataset
bitcoin_blockchain.list_tables()
# print information on all the columns in the "full" table
# in the hacker_news dataset
bitcoin_blockchain.table_schema("blocks")