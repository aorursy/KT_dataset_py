# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



from bq_helper import BigQueryHelper



# Helper object for BigQuery Ethereum dataset

btc = BigQueryHelper(active_project="bigquery-public-data", 

                     dataset_name="cyrpto_bitcoin",

                     max_wait_seconds=1800)
query = """

WITH blockOutputs AS (

    SELECT 

      tx.block_number AS block

      ,ROUND((SUM(tx.value) / 100000000), 3) AS amount_sent

    FROM `bigquery-public-data.crypto_bitcoin.outputs` tx

    GROUP BY tx.block_number

    ORDER BY tx.block_number ASC

),



blockInputs AS (

    SELECT

      tx.block_number AS block

      ,ROUND((SUM(tx.value) / 100000000), 3) AS amount_received

    FROM `bigquery-public-data.crypto_bitcoin.inputs` tx

    GROUP BY tx.block_number

    ORDER BY tx.block_number ASC

),



txFees AS (

    SELECT

      tx.block_number AS block

      ,ROUND((SUM(tx.fee) / 100000000), 3) AS amount_fee

    FROM `bigquery-public-data.crypto_bitcoin.transactions` tx

    GROUP BY tx.block_number

    ORDER BY tx.block_number ASC

),



-- Compute each block's flows

blockBalances AS (

    SELECT r.block,

           SUM(r.amount_received) AS received,

           SUM(s.amount_sent) AS sent,

           SUM(f.amount_fee) AS fee

    FROM blockInputs AS r, blockOutputs AS s, txFees AS f

    WHERE r.block = s.block AND s.block = f.block

    GROUP BY r.block

)



-- Compute a rolling sum across all blocks

SELECT bb.block, 

       bd.timestamp,

       sum(bb.received) 

           OVER (ORDER BY block ROWS UNBOUNDED PRECEDING) AS total_received,

       sum(bb.sent) 

           OVER (ORDER BY block ROWS UNBOUNDED PRECEDING) AS total_sent,

       sum(bb.fee) 

           OVER (ORDER BY block ROWS UNBOUNDED PRECEDING) AS total_fee

FROM blockBalances bb,

     `bigquery-public-data.crypto_bitcoin.blocks` bd

WHERE bb.block = bd.number

ORDER BY block ASC

"""



# Estimate how big this query will be

btc.estimate_query_size(query)
# Store the results into a Pandas DataFrame

df = btc.query_to_pandas_safe(query, max_gb_scanned=40)
df.head()
import matplotlib.pyplot as plt
df2 = df.copy()

df2['sent'] = df['total_sent'].apply(float).apply(np.log10)

df2['received'] = df['total_received'].apply(float).apply(np.log10)

df2['fees'] = df['total_fee'].apply(float).apply(float).apply(np.log10)

df2['net'] = df2['received'] - df2['sent']



plt.figure(figsize=(16, 9))



plt.plot('block', 'sent', data=df2)

plt.plot('block', 'received', data=df2)

plt.plot('block', 'fees', data=df2)

plt.plot('block', 'net', data=df2)



plt.title('BTC circulating supply by flow type')

plt.legend()
plt.figure(figsize=(16, 9))





plt.plot('block', 'net', data=df2)



plt.title('BTC circulating supply (net only)')

plt.legend()