# setup
import bq_helper

bitcoin = bq_helper.BigQueryHelper(active_project='bigquery-public-data',
                              dataset_name='bitcoin_blockchain')

bitcoin.list_tables()
query1 = '''WITH t AS(
        SELECT TIMESTAMP_MILLIS(timestamp) AS transaction_time, transaction_id
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        WHERE EXTRACT(YEAR FROM (TIMESTAMP_MILLIS(timestamp))) = 2017
        )
        SELECT DATE(EXTRACT(YEAR FROM transaction_time), EXTRACT(MONTH FROM transaction_time), EXTRACT(DAY FROM transaction_time)) AS date, COUNT(transaction_id) AS transactions
        FROM t
        GROUP BY 1
        ORDER BY 1
        '''
result1= bitcoin.query_to_pandas(query1)
result1
import matplotlib.pyplot as plt

ax = plt.subplots(figsize=(15,7))
plt.plot(result1.date, result1.transactions)
plt.title('2017 Bitcoin Transactions')
query2 = '''SELECT merkle_root, COUNT(transaction_id) AS transactions
        FROM `bigquery-public-data.bitcoin_blockchain.transactions`
        GROUP BY 1
        ORDER BY 2 DESC
        '''
result2 = bitcoin.query_to_pandas(query2)
result2
