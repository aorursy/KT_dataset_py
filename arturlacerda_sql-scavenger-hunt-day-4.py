import bq_helper
import seaborn as sns
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.head('transactions')
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(DATE FROM trans_time) AS date
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY date
            ORDER BY date
        """
transactions_by_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
transactions_by_day.head()
transactions_by_day.plot(x='date', y='transactions', figsize=(15,10))
query = """ SELECT COUNT(transaction_id) as num_transactions, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY num_transactions DESC
        """
transactions_by_merkle_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=37)
transactions_by_merkle_root
sns.distplot(transactions_by_merkle_root.num_transactions)