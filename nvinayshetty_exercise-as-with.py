# import package with helper functions 
import bq_helper

# create a helper object for this dataset
bitcoin_blockchain = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                              dataset_name="bitcoin_blockchain")
# Your Code Here
query= """ WITH dailytranscation AS
            (
            SELECT TIMESTAMP_MILLIS(timestamp) as datetime,count(transaction_id) AS transaction
            FROM `bigquery-public-data.bitcoin_blockchain.transactions`
            GROUP BY datetime
            )
            SELECT EXTRACT(DAY FROM datetime) AS day,
                   EXTRACT(MONTH FROM datetime) AS month, 
                   EXTRACT(YEAR FROM datetime) AS year,
                   count(transaction) AS transactions_per_day
                   FROM dailytranscation 
                   GROUP BY day,month,year
                   HAVING year=2017
                   ORDER BY day,month

       """
bitcoin_blockchain.query_to_pandas_safe(query,max_gb_scanned=25)
# Your Code Here
