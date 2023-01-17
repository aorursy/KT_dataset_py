# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
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
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(transactions_per_month.transactions)
plt.title("Monthly Bitcoin Transcations")
query = ''' WITH time AS
(
    SELECT TIMESTAMP_MILLIS(timestamp) AS date,
        transaction_id
    FROM `bigquery-public-data.bitcoin_blockchain.transactions`
)
SELECT COUNT(transaction_id) AS transactions,
    EXTRACT(DATE from date) AS date
FROM time
GROUP BY date
ORDER BY date
        '''
transactions_per_day = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

transactions_per_day
bitcoin_blockchain.head('blocks')
#* How many blocks are associated with each merkle root?
#    * You can use the "merkle_root" and "block_id" columns in the "blocks"
#table to answer this question.  
query = ''' WITH blah AS
(
    SELECT block_id AS b_id,
        merkle_root AS root
    FROM `bigquery-public-data.bitcoin_blockchain.blocks`
)
SELECT COUNT(b_id) AS blocks,
    root
FROM blah
GROUP BY root
ORDER BY blocks DESC
        '''
blocks_per_root = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)
blocks_per_root