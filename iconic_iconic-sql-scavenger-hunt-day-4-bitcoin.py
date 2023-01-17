import bq_helper

bc = bq_helper.BigQueryHelper(active_project = "bigquery-public-data",
                              dataset_name   = "bitcoin_blockchain")
query = """WITH daytrx AS
                (SELECT EXTRACT(DATE FROM TIMESTAMP_MILLIS(timestamp)) as dte,
                        transaction_id as trx
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                 WHERE EXTRACT(YEAR FROM TIMESTAMP_MILLIS(timestamp)) = 2017
                )
            
            SELECT dte, count(trx) as cnt_transactions
            FROM daytrx
            GROUP BY dte
            ORDER BY dte asc
        """

trx_per_day = bc.query_to_pandas_safe(query, max_gb_scanned=30)

trx_per_day
# import plotting library
import matplotlib.pyplot as plt

# plot monthly bitcoin transactions
plt.plot(trx_per_day.cnt_transactions)
plt.title("Daily Bitcoin Transcations")
query = """WITH bm AS
           (SELECT block_id as block, merkle_root as merkle
            FROM `bigquery-public-data.bitcoin_blockchain.blocks`
           )
           
           SELECT cnt_block, count(merkle) as cnt_recs
           FROM 
           (
               SELECT merkle, count(block) as cnt_block
               FROM bm
               GROUP BY merkle
           )
           GROUP BY cnt_block
        """

blocks_per_merkle = bc.query_to_pandas_safe(query, max_gb_scanned=30)

blocks_per_merkle
