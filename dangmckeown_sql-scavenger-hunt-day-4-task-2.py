# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")



query = """ SELECT merkle_root, COUNT(transaction_id) as count_of_transactions 
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY count_of_transactions DESC
        """

transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas(query)

transactions_by_merkle_root