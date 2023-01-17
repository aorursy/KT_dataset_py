import bq_helper
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                                             dataset_name= "bitcoin_blockchain")
query = """ WITH time AS 
            (
                SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time,
                    transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT(MONTH FROM trans_time) AS month,
                EXTRACT(YEAR FROM trans_time) AS year
            FROM time
            GROUP BY year, month 
            ORDER BY year, month
        """
# note that max_gb_scanned is set to 21, rather than 1

transactions_per_month = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
query2 = """WITH time_day AS
            (
            SELECT TIMESTAMP_MILLIS(timestamp)AS trans_time,
                transaction_id
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
            )
            
            SELECT COUNT(transaction_id) AS transactions,
                EXTRACT (DAYOFYEAR FROM trans_time) AS day,
                EXTRACT (MONTH FROM trans_time) AS month
                
                FROM time_day
                GROUP BY day,month
                ORDER BY day,month
                """

transactions_per_year = bitcoin_blockchain.query_to_pandas_safe(query2,max_gb_scanned=21)
print(transactions_per_year)
query3 = """WITH time_trans AS
           (
               SELECT merkle_root AS mk, 
               transaction_id
                FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            )
              SELECT mk,
                  COUNT(transaction_id) AS transactions
                                             
              FROM time_trans
              GROUP by mk
              ORDER by transactions
            """
tran_by_merkle = bitcoin_blockchain.query_to_pandas_safe(query3,max_gb_scanned=38)
print(tran_by_merkle)