# Importing helper package
import bq_helper

#Creating helper Object with this dataset
btc_blockchain=bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                                        dataset_name='bitcoin_blockchain')
#query for returning transcations per day
query1='''WITH time AS(SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
                    )
                    SELECT COUNT(transaction_id) AS nbr_of_transactions,
                           EXTRACT(Day from trans_time) as Day,
                           EXTRACT(Year from trans_time) as Year
                           FROM time
                           WHERE EXTRACT(Year from trans_time)=2017
                           GROUP BY 3,2'''

transcation_per_day=btc_blockchain.query_to_pandas_safe(query1,max_gb_scanned=21)

import matplotlib.pyplot as plb

plb.plot(transcation_per_day.nbr_of_transactions)
plb.title("Daily Bitcoin Transcations")
query2='''SELECT COUNT(block_id) AS Nbr_of_Block_Id,
                 merkle_root
          FROM `bigquery-public-data.bitcoin_blockchain.blocks`
          GROUP BY 2
          Order BY 1 DESC
       '''

blocks_per_merkle=btc_blockchain.query_to_pandas_safe(query2)
import matplotlib.pyplot as plb

plb.plot(blocks_per_merkle.Nbr_of_Block_Id)
plb.title("Number of blocks per Merkle")

