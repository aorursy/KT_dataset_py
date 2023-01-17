









# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")

#query to see how many bitcoins transactions were made each day in 2017
#Using the 'transactions' table
query = """WITH time AS
            (  
                 SELECT TIMESTAMP_MILLIS(timestamp) AS trans_time, transaction_id
                 FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            
            )
            SELECT COUNT(transaction_id) AS transactions,
                   EXTRACT (DAY FROM trans_time) AS day,
                   EXTRACT (MONTH FROM trans_time) AS month
            FROM time
            WHERE EXTRACT(YEAR FROM trans_time) = 2017
            GROUP BY month,day
            ORDER BY month,day
            """

#Estimating the query size
print(bitcoin_blockchain.estimate_query_size(query))
#max_GB scanned to set to 21 rather than default 1GB
trans_per_day_2017 = bitcoin_blockchain.query_to_pandas_safe(query, max_gb_scanned=21)

#display the results
trans_per_day_2017
#Frequency plot - to see the result data visually for better understanding
import matplotlib.pyplot as plt
plt.plot(trans_per_day_2017)
plt.xlabel("Days")
plt.ylabel("Frequency")
plt.title("Daily Bitcoin Transaction in 2017")

#query to see how many transactions are associated with each merkle root
#Using 'transactions' table
query1 = """SELECT COUNT(transaction_id) AS transactions_count, merkle_root
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY merkle_root
            ORDER BY transactions_count DESC
         """

#Estimating the query size
print(bitcoin_blockchain.estimate_query_size(query1))
#max_GB scanned is set to 38 rather than default 1 GB
transaction_per_merkel =  bitcoin_blockchain.query_to_pandas_safe(query1, max_gb_scanned=38)

#display the results
transaction_per_merkel
