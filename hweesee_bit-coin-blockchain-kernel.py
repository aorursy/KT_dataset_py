# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

# print all the tables in this dataset
bitcoin_blockchain.list_tables()
# print the first couple rows of the "blocks" dataset
bitcoin_blockchain.head("blocks")
# print the first couple rows of the "transactions" dataset
bitcoin_blockchain.head("transactions")
query_top10= """ 
            WITH TRX AS 
            (
                SELECT 
                block_id, COUNT(transaction_id)  AS transactions_count
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY block_id
                --ORDER BY transactions_count DESC LIMIT 10
            )
            SELECT block_id, transactions_count
            FROM TRX
            ORDER BY transactions_count DESC LIMIT 10            
        """

# check how big this query will be
bitcoin_blockchain.estimate_query_size(query_top10)
# note that max_gb_scanned is set to 37, rather than 1
transactions_per_top10block = bitcoin_blockchain.query_to_pandas_safe(query_top10, max_gb_scanned=37)
transactions_per_top10block
query_trxdate = """ 
            WITH TRX AS 
            (
                SELECT 
                block_id, COUNT(transaction_id)  AS transactions_count
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                GROUP BY block_id
                ORDER BY transactions_count DESC LIMIT 10
            )
            SELECT 
                topblock.block_id,
                topblock.transactions_count,
                MIN(TIMESTAMP_MILLIS(trans.timestamp)) AS earliest_trx,
                MAX(TIMESTAMP_MILLIS(trans.timestamp)) AS latest_trx
            FROM 
            TRX AS topblock
            INNER JOIN `bigquery-public-data.bitcoin_blockchain.transactions` AS trans
                ON trans.block_id = topblock.block_id
            GROUP BY topblock.block_id, topblock.transactions_count
            ORDER BY topblock.transactions_count DESC
        """
# check how big this query will be
bitcoin_blockchain.estimate_query_size(query_trxdate)
# note that max_gb_scanned is set to 40, rather than 1
transactionsanddate_per_top10block = bitcoin_blockchain.query_to_pandas_safe(query_trxdate, max_gb_scanned=40)
transactionsanddate_per_top10block
# save our dataframe as a .csv 
transactionsanddate_per_top10block.to_csv("top10block_bitcoin_transactions.csv")