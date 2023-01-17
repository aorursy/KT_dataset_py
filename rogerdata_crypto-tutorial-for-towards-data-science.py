from google.cloud import bigquery

client = bigquery.Client()
from google.cloud import bigquery

import pandas as pd



client = bigquery.Client()



query = """

#standardSQL

    SELECT

    COUNT(block_id)

    FROM `bigquery-public-data.bitcoin_blockchain.blocks`

"""



query_job = client.query(query)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

transactions = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the first row, which will contain the count. 

transactions.head(1)
aggquery = """

WITH last_year AS (

  SELECT transaction_id, DATE(TIMESTAMP_MILLIS(timestamp)) as DAY

  FROM `bigquery-public-data.bitcoin_blockchain.transactions`

  WHERE DATE(TIMESTAMP_MILLIS(timestamp)) > DATE_SUB(DATE(2018,9,10), INTERVAL 1 YEAR))

SELECT DAY, COUNT(transaction_id) AS COUNT_OF_TRANSACTIONS

FROM last_year

GROUP BY DAY

ORDER BY COUNT_OF_TRANSACTIONS DESC

"""
client = bigquery.Client()



# Taking the query from the previous cell. 



query_job = client.query(aggquery)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

transactionsum = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the top ten rows in the dataframe, which will contain the top ten results. 

transactionsum.head(10)
client = bigquery.Client()



dayquery = """

WITH last_year AS (

  SELECT transaction_id, DATE(TIMESTAMP_MILLIS(timestamp)) as DAY

  FROM `bigquery-public-data.bitcoin_blockchain.transactions`

  WHERE DATE(TIMESTAMP_MILLIS(timestamp)) > DATE_SUB(DATE(2018,9,10), INTERVAL 1 YEAR))

SELECT DAY, COUNT(transaction_id) AS COUNT_OF_TRANSACTIONS

FROM last_year

GROUP BY DAY

ORDER BY DAY DESC

"""



query_job = client.query(dayquery)



iterator = query_job.result(timeout=30)

rows = list(iterator)



# Transform the rows into a nice pandas dataframe

transactionsday = pd.DataFrame(data=[list(x.values()) for x in rows], columns=list(rows[0].keys()))



# Look at the ten top rows as a preview

transactionsday.head(10)
import matplotlib

from matplotlib import pyplot as plt

%matplotlib inline



transactionsday.plot()