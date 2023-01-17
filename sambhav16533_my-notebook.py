import bq_helper

bitcoin_blockchain = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.table_schema("blocks")
bitcoin_blockchain.head("blocks")
bitcoin_blockchain.head("blocks", selected_columns="block_id", num_rows=5)
query1 = """SELECT * 
            FROM bigquery-public-data.bitcoin_blockchain.blocks"""
bitcoin_blockchain.estimate_query_size(query1)
