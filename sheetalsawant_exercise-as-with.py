# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
bitcoin_blockchain.list_tables()
bitcoin_blockchain.head("transactions")
query = """SELECT COUNT(transaction_id) AS transaction, 
           TIMESTAMP_MILLIS(timestamp) AS trans_time
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY trans_time
        """
bitcoin_blockchain.query_to_pandas(query)
# Your Code Here
query = """WITH time AS
           (SELECT COUNT(transaction_id) AS transaction, 
           TIMESTAMP_MILLIS(timestamp) AS trans_time
           FROM `bigquery-public-data.bitcoin_blockchain.transactions`
           GROUP BY trans_time
           )
           SELECT EXTRACT(DAY FROM trans_time) AS day,
                  EXTRACT(MONTH FROM trans_time) AS month,
                  EXTRACT(YEAR FROM trans_time) AS year,
                  COUNT(transaction) AS no_of_transactions
                  FROM time
                  GROUP BY day,month,year
                  HAVING year=2017
                  ORDER BY month,day 
        """
bitcoin_blockchain.query_to_pandas(query)
# Your Code Here
query=""" WITH merkle_each AS
          (SELECT transaction_id, merkle_root AS merkle
          FROM `bigquery-public-data.bitcoin_blockchain.transactions`
          )
          SELECT COUNT(transaction_id) AS transaction, merkle
          FROM merkle_each
          GROUP BY merkle
          ORDER BY transaction DESC
        """
bitcoin_blockchain.query_to_pandas(query)