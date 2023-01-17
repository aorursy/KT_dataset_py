import bq_helper
bitcoin_bc = bq_helper.BigQueryHelper(active_project= "bigquery-public-data", 
                                       dataset_name = "bitcoin_blockchain")
bitcoin_bc.list_tables()
bitcoin_bc.table_schema("transactions")
bitcoin_bc.head("transactions")
bitcoin_bc.head("transactions", selected_columns="timestamp", num_rows=10)
query = """SELECT work_error
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            where timestamp<=1515249"""
bitcoin_bc.estimate_query_size(query)
