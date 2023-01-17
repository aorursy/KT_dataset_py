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
# preview the first couple lines of the "full" table
bitcoin_blockchain.head("blocks")
bitcoin_blockchain.head("blocks", selected_columns="timestamp", num_rows=10)
query = """SELECT version
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
            WHERE timestamp = 1502253852000 """
bitcoin_blockchain.estimate_query_size(query)
#Running this query will take around 7 MB
#import pandas
# only run this query if it's less than 0.000001 GB
bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=0.0000001)
df_prova=bitcoin_blockchain.query_to_pandas(query)
df_prova=bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=0.1)
#df_prova.head()
df_prova.version.mean()
df_prova.to_csv("prova1.csv")
